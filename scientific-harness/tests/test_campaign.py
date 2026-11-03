"""Check fitting, budget holds, single-controller locks, restarts, and saved selection."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock
from unittest.mock import Mock

import pytest

from cases.pressure import PressureCase
from controller.artifacts import read_json
from controller.campaign import Campaign
from controller.experiment import evaluate_selection, prepare_case
from controller.hypotheses import Hypothesis
from numerical.geology import BandGeometry


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    case = PressureCase(grid_size=8, segment_steps=4, sample_every=1)
    prepare_case(
        tmp_path / "snapshot",
        case,
        truth_geometry=BandGeometry(0.5, 0.25),
    )
    return tmp_path / "snapshot" / "evidence"


def band(orientation: str = "vertical", center: float = 0.5) -> dict[str, object]:
    return {
        "kind": "band",
        "orientation": orientation,
        "center_fraction": center,
        "width_fraction": 0.25,
    }


@pytest.mark.parametrize(
    ("members", "max_evaluations", "max_simulations"),
    [(50, 3, 750), (16, 2, 160), (8, 1, 40), (16, 33, 2640), (8, 1000, 40000)],
)
def test_simulation_budget_is_derived_from_members_and_evaluation_limit(
    tmp_path: Path, evidence: Path, members: int, max_evaluations: int, max_simulations: int
) -> None:
    output = tmp_path / "campaign"
    with Campaign(evidence, output, members=members, max_evaluations=max_evaluations) as campaign:
        state = campaign.snapshot()
        assert campaign.planned_calls == 5 * members
        assert campaign.max_simulations == max_simulations
        assert state["reserved_simulator_calls"] == 5 * members
        assert state["remaining_simulator_calls"] == max_simulations - 5 * members
        assert state["remaining_evaluations"] == max_evaluations - 1
        assert read_json(output / "campaign.json")["max_simulations"] == max_simulations


@pytest.mark.parametrize(
    "case",
    [
        PressureCase(grid_size=66),
        PressureCase(segment_steps=66),
        PressureCase(segment_steps=34, sample_every=1),
    ],
    ids=["grid", "time_steps", "conditioning_samples"],
)
def test_local_resource_limits_reject_oversized_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: PressureCase
) -> None:
    monkeypatch.setattr("controller.campaign.load_case", Mock(return_value=(case, {})))
    output = tmp_path / "campaign"
    with pytest.raises(ValueError, match="laptop controller"):
        Campaign(tmp_path / "evidence", output)
    assert not output.exists()


def test_initial_failure_can_be_repaired_and_graded_only_after_freezing(
    tmp_path: Path, evidence: Path
) -> None:
    output = tmp_path / "campaign"
    with Campaign(evidence, output, members=16) as campaign:
        initial = campaign.snapshot()["evaluations"][0]
        assert initial["hypothesis"] == {"kind": "homogeneous"}
        assert initial["status"] == "queued"
        completed = campaign.execute(initial["run_id"])
        assert completed["result"]["adequacy_gate_passed"] is False
        assert campaign.execute(initial["run_id"]) == completed
        assert campaign.snapshot()["charged_simulator_calls"] == 80
        with pytest.raises(ValueError, match="pressure-error check"):
            campaign.finish("adequate", "Pretend the homogeneous model is good")
        repaired = campaign.submit(band(), "Test a barrier separating east and west")
        campaign.execute(repaired["run_id"])
        decision = campaign.finish(
            "adequate", "The revised model passes the fixed pressure-error check"
        )
        assert decision["closed"]
        assert len(decision["evaluations"]) == 2
        assert decision["charged_simulator_calls"] == 160
        assert decision["model_calls"] == 0
        with pytest.raises(ValueError, match="closed"):
            campaign.submit(band("horizontal"), "Try another recipe after the decision")
    report = evaluate_selection(evidence, evidence.parent / "protected", output, tmp_path / "grade")
    assert report["structural_truth_recovered"] is True
    assert report["selected_candidate_id"] == repaired["run_id"]


def test_atomic_reservation_blocks_combined_over_budget_parallel_requests(
    tmp_path: Path, evidence: Path
) -> None:
    with Campaign(evidence, tmp_path / "campaign", members=16, max_evaluations=2) as campaign:

        def submit(orientation: str) -> bool:
            try:
                campaign.submit(band(orientation), "Parallel proposal")
                return True
            except ValueError as error:
                assert str(error) == "hypothesis-evaluation budget exhausted"
                return False

        with ThreadPoolExecutor(max_workers=2) as executor:
            accepted = list(executor.map(submit, ["vertical", "horizontal"]))
        assert sum(accepted) == 1
        state = campaign.snapshot()
        assert state["reserved_simulator_calls"] == 160
        assert state["remaining_simulator_calls"] == 0
        assert state["remaining_evaluations"] == 0
        assert len(state["evaluations"]) == 2


def test_direct_clients_cannot_execute_two_candidates_at_once(
    tmp_path: Path, evidence: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_started = Event()
    second_started = Event()
    release_first = Event()
    entry_lock = Lock()
    entries = 0

    def condition(hypothesis: Hypothesis, *_arguments: object) -> dict[str, object]:
        nonlocal entries
        with entry_lock:
            entries += 1
            first = entries == 1
        if first:
            first_started.set()
            assert release_first.wait(timeout=10)
        else:
            second_started.set()
        return {
            "family": hypothesis.family.value,
            "hypothesis": hypothesis.specification(),
            "status": "failed",
            "error": "Execution-policy fixture; no numerical work attempted",
            "simulator_calls": 0,
        }

    monkeypatch.setattr("controller.campaign.condition_candidate", condition)
    with Campaign(evidence, tmp_path / "campaign", members=8, max_evaluations=2) as campaign:
        initial = campaign.snapshot()["evaluations"][0]
        proposed = campaign.submit(band(), "Concurrent direct-client request")
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(campaign.execute, initial["run_id"])
            try:
                assert first_started.wait(timeout=10)
                second = workers.submit(campaign.execute, proposed["run_id"])
                assert not second_started.wait(timeout=0.25)
                assert campaign.evaluation(proposed["run_id"])["status"] == "queued"
            finally:
                release_first.set()
            assert first.result(timeout=10)["status"] == "failed"
            assert second.result(timeout=10)["status"] == "failed"
        assert second_started.is_set()
        assert campaign.snapshot()["reserved_simulator_calls"] == 0
        assert campaign.snapshot()["charged_simulator_calls"] == 0


def test_equivalent_requests_do_not_reserve_or_execute_another_batch(
    tmp_path: Path, evidence: Path
) -> None:
    with Campaign(evidence, tmp_path / "campaign", members=16) as campaign:
        first = campaign.submit(band(), "First proposal")
        repeated = campaign.submit(
            band(center=0.499), "Different center value covering the same grid cells"
        )
        assert repeated["run_id"] == first["run_id"]
        assert len(campaign.snapshot()["evaluations"]) == 2
        assert campaign.snapshot()["reserved_simulator_calls"] == 160
        with pytest.raises(ValueError, match="wait"):
            campaign.finish("uncertain", "Finish while accepted work is pending")


def test_public_context_excludes_truth_geometry_seeds_paths_and_final_observations(
    tmp_path: Path, evidence: Path
) -> None:
    with Campaign(evidence, tmp_path / "campaign", members=16) as campaign:
        public = campaign.public_evidence()
        fixed_inference = public["hypothesis_grammar"]["fixed_inference"]
        assert public["physics"] == asdict(campaign.case)
        assert "band_width_fraction" not in public["physics"]
        assert "truth_family" not in public
        assert "noise_seed" not in public
        assert "implementation" not in public
        assert "final_observations" not in public
        assert "fractions of the domain length" in public["hypothesis_grammar"]["geometry_rule"]
        assert "16 ensemble members; 4 equal-inflation ES-MDA updates" in fixed_inference
        assert "fixed priors" in fixed_inference
        assert "conditioning and validation RMSE" in public["adequacy_rule"]
        assert "not proof of geological truth" in public["adequacy_rule"]
        assert len(public["times_hours"]) * len(public["gauge_ids"]) == (
            campaign.conditioning["values_pa"].size
        )
        (evidence.parent / "protected").rename(tmp_path / "operator_only")
        initial = campaign.snapshot()["evaluations"][0]
        campaign.execute(initial["run_id"])
        assert campaign.snapshot()["evaluations"][0]["result"]["diagnostics"]["signed_residual_std"]


def test_truth_geometry_does_not_change_public_settings_or_hypothesis_grammar(
    tmp_path: Path,
) -> None:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    public = []
    for index, geometry in enumerate((BandGeometry(0.5, 0.25), BandGeometry(0.625, 0.5))):
        snapshot = tmp_path / f"snapshot-{index}"
        prepare_case(snapshot, case, truth_geometry=geometry)
        manifest = read_json(snapshot / "evidence" / "evidence.json")
        assert manifest["case"] == case.specification()
        assert "band_width_fraction" not in manifest["case"]["settings"]
        with Campaign(snapshot / "evidence", tmp_path / f"campaign-{index}", members=2) as campaign:
            public.append(campaign.public_evidence())
    first, second = public
    assert first["physics"] == second["physics"] == asdict(case)
    assert first["hypothesis_grammar"] == second["hypothesis_grammar"]
    assert first["observed_pressure_change_bar"] != second["observed_pressure_change_bar"]


def test_model_requests_have_independent_persisted_cost_and_call_limits(
    tmp_path: Path, evidence: Path
) -> None:
    with Campaign(
        evidence,
        tmp_path / "campaign",
        members=16,
        max_model_calls=2,
        max_model_cost_usd=1.0,
    ) as campaign:
        reservation = {
            "call_id": "mc_" + "a" * 32,
            "reservation_usd": 0.8,
            "provider": "faux",
            "model": "fixture",
        }
        campaign.reserve_model_call(reservation)
        with pytest.raises(ValueError, match="already reserved"):
            campaign.reserve_model_call(reservation)
        with pytest.raises(ValueError, match="budget exhausted"):
            campaign.reserve_model_call({**reservation, "call_id": "mc_" + "b" * 32})
        campaign.settle_model_call({"call_id": reservation["call_id"], "cost_usd": 0.1})
        campaign.reserve_model_call({**reservation, "call_id": "mc_" + "b" * 32})
        campaign.settle_model_call({"call_id": "mc_" + "b" * 32, "cost_usd": None})
        assert campaign.snapshot()["model_calls"] == 2
        assert campaign.snapshot()["model_cost_estimate_usd"] == pytest.approx(0.9)
        with pytest.raises(ValueError, match="budget exhausted"):
            campaign.reserve_model_call(
                {**reservation, "call_id": "mc_" + "c" * 32, "reservation_usd": 0.0}
            )
        campaign.execute(campaign.snapshot()["evaluations"][0]["run_id"])
        campaign.finish("uncertain", "Model-budget fixture ended before further proposals")
    report = evaluate_selection(
        evidence, evidence.parent / "protected", tmp_path / "campaign", tmp_path / "grade"
    )
    assert report["model_calls"] == 2
    assert report["model_cost_estimate_usd"] == pytest.approx(0.9)


def test_one_controller_owns_recovery_and_interrupted_work_is_not_replayed(
    tmp_path: Path, evidence: Path
) -> None:
    output = tmp_path / "campaign"
    with Campaign(evidence, output, members=16) as campaign:
        with pytest.raises(ValueError, match="another controller"):
            Campaign(evidence, output, members=16)
        identifier = campaign.snapshot()["evaluations"][0]["run_id"]
        with closing(campaign.connect()) as connection, connection:
            connection.execute(
                "UPDATE evaluations SET status = 'running' WHERE id = ?", (identifier,)
            )
    with Campaign(evidence, output, members=16) as restarted:
        state = restarted.snapshot()
        assert state["evaluations"][0]["status"] == "interrupted"
        assert state["charged_simulator_calls"] == 80
        assert state["reserved_simulator_calls"] == 0
        assert restarted.execute(identifier)["status"] == "interrupted"
        assert not (output / identifier / "posterior.npz").exists()


def test_finished_search_remains_discoverable_without_new_simulations(
    tmp_path: Path, evidence: Path
) -> None:
    output = tmp_path / "campaign"
    with Campaign(evidence, output, members=16) as campaign:
        initial = campaign.snapshot()["evaluations"][0]
        campaign.execute(initial["run_id"])
        campaign.finish("uncertain", "No adequate model established in this fixture")
    selection = read_json(output / "selection.json")
    with Campaign(evidence, output, members=16) as restarted:
        assert restarted.snapshot()["closed"]
        assert read_json(output / "selection.json") == selection


def test_constructed_shifted_band_is_not_limited_to_the_original_three_templates(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    prepare_case(snapshot, PressureCase(), truth_geometry=BandGeometry(0.65, 0.2))
    output = tmp_path / "campaign"
    with Campaign(snapshot / "evidence", output) as campaign:
        campaign.execute(campaign.snapshot()["evaluations"][0]["run_id"])
        wrong = campaign.submit(
            {
                "kind": "band",
                "orientation": "vertical",
                "center_fraction": 0.5,
                "width_fraction": 0.1,
            },
            "Try the original template",
        )
        campaign.execute(wrong["run_id"])
        corrected = campaign.submit(
            {
                "kind": "band",
                "orientation": "vertical",
                "center_fraction": 0.65,
                "width_fraction": 0.2,
            },
            "Revise the geometry based on residuals",
        )
        result = campaign.execute(corrected["run_id"])
        assert result["result"]["adequacy_gate_passed"] is True
        campaign.finish(
            "adequate", "The proposed strip position and width pass the pressure-error check"
        )
    report = evaluate_selection(
        snapshot / "evidence", snapshot / "protected", output, tmp_path / "grade"
    )
    assert report["structural_truth_recovered"] is True
    assert report["selected_candidate_id"] == corrected["run_id"]
