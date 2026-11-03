"""Test campaign recovery, protected grading, saved updates, and input checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from cases.pressure import Phase, PressureCase
from controller.artifacts import checksum, read_arrays, read_json
from controller.assimilation import create_smoother, update_parameters
from controller.campaign import Campaign
from controller.experiment import (
    evaluate_selection,
    load_case,
    load_observations,
    load_posterior,
    prepare_case,
)
from controller.hypotheses import parse_hypothesis
from numerical.geology import (
    BandGeometry,
    Family,
    parameter_names,
    permeability_field,
    sample_prior,
)
from numerical.pressure import PressureModel


@pytest.fixture
def small_snapshot(tmp_path: Path) -> tuple[PressureCase, Path]:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    root = tmp_path / "snapshot"
    prepare_case(
        root,
        case,
        noise_seed=101,
        truth_geometry=BandGeometry(0.5, 0.25),
    )
    return case, root


def band(width_fraction: float, orientation: str = "vertical") -> dict[str, object]:
    return {
        "kind": "band",
        "orientation": orientation,
        "center_fraction": 0.5,
        "width_fraction": width_fraction,
    }


@pytest.fixture
def small_campaign(
    tmp_path: Path, small_snapshot: tuple[PressureCase, Path]
) -> tuple[PressureCase, Path, Path]:
    case, snapshot = small_snapshot
    run = tmp_path / "campaign"
    with Campaign(snapshot / "evidence", run, members=8, max_evaluations=2) as campaign:
        campaign.execute(campaign.snapshot()["evaluations"][0]["run_id"])
        proposed = campaign.submit(band(0.25), "Scripted numerical regression fixture")
        campaign.execute(proposed["run_id"])
        campaign.finish("uncertain", "Numerical fixture, not an LLM discovery result")
    return case, snapshot, run


def test_preparation_validates_truth_geometry_separately_from_case_settings(tmp_path: Path) -> None:
    case = PressureCase(grid_size=8)
    snapshot = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="band width must lie"):
        prepare_case(snapshot, case, truth_geometry=BandGeometry(0.5, 0.1))
    assert not snapshot.exists()


@pytest.mark.parametrize("seed", [7, 11, 19])
@pytest.mark.parametrize(
    ("family", "background_md", "band_ratio"),
    [(Family.VERTICAL_BAND, 100.0, 0.01), (Family.HORIZONTAL_BAND, 80.0, 0.02)],
)
def test_campaign_recovers_two_truths_across_three_seeds(
    tmp_path: Path, seed: int, family: Family, background_md: float, band_ratio: float
) -> None:
    case = PressureCase()
    snapshot = tmp_path / "snapshot"
    run = tmp_path / "campaign"
    prepare_case(
        snapshot,
        case,
        family,
        background_md,
        band_ratio,
        noise_seed=2026 + seed,
        truth_geometry=BandGeometry(0.5, 0.1),
    )
    with Campaign(snapshot / "evidence", run, seed=seed) as campaign:
        campaign.execute(campaign.snapshot()["evaluations"][0]["run_id"])
        for orientation in ("vertical", "horizontal"):
            proposed = campaign.submit(
                band(0.1, orientation),
                "Scripted hypothesis for numerical recovery and validation-ranking checks",
            )
            campaign.execute(proposed["run_id"])
        campaign.finish("uncertain", "Numerical fixture, not an LLM discovery result")
    selection = read_json(run / "selection.json")
    selection_digest = checksum(run / "selection.json")
    report = evaluate_selection(
        snapshot / "evidence", snapshot / "protected", run, tmp_path / "evaluation"
    )
    assert report["family_recovered"] is True
    assert report["structural_truth_recovered"] is True
    assert report["proposer_mode"] == "no_model_calls"
    assert report["parameter_recovery"]["background_md"]["relative_median_error"] < 0.15
    assert report["parameter_recovery"]["band_ratio"]["relative_median_error"] < 0.35
    assert selection["final_evaluation_accessed"] is False
    assert selection["simulator_calls"] <= 750
    assert checksum(run / "selection.json") == selection_digest
    assert (tmp_path / "evaluation" / "diagnostics.png").read_bytes().startswith(b"\x89PNG")
    selected = next(
        record for record in selection["candidates"] if record["family"] == family.value
    )
    assert selected["conditioning_rmse_final_pa"] < selected["conditioning_rmse_history_pa"][0]
    posterior = load_posterior(run, case, selected)
    assert posterior["parameters"].shape == (2, 50)
    assert posterior["responses_pa"].shape == (128, 50)


def test_campaign_uses_no_protected_files_and_can_repeat_exactly(
    tmp_path: Path, small_snapshot: tuple[PressureCase, Path]
) -> None:
    case, snapshot = small_snapshot
    evidence = snapshot / "evidence"
    public_manifest = (evidence / "evidence.json").read_text()
    assert "truth_family" not in public_manifest
    assert "band_width_fraction" not in public_manifest
    assert "noise_seed" not in public_manifest
    assert "vertical_band" not in public_manifest
    (snapshot / "protected").rename(tmp_path / "operator_only")
    selections: list[dict[str, object]] = []
    for run in (tmp_path / "first", tmp_path / "second"):
        with Campaign(evidence, run, members=8, max_evaluations=2) as campaign:
            campaign.execute(campaign.snapshot()["evaluations"][0]["run_id"])
            proposed = campaign.submit(
                band(0.25),
                "Repeat the same numerical fixture without protected data",
            )
            campaign.execute(proposed["run_id"])
            campaign.finish("uncertain", "Numerical fixture, not an LLM discovery result")
        selections.append(read_json(run / "selection.json"))
    first, second = selections
    assert first["selected_family"] == second["selected_family"]
    assert first["simulator_calls"] == second["simulator_calls"]
    for first_candidate, second_candidate in zip(
        first["candidates"], second["candidates"], strict=True
    ):
        assert first_candidate["status"] == second_candidate["status"]
        if first_candidate["status"] == "succeeded":
            first_arrays = load_posterior(tmp_path / "first", case, first_candidate)
            second_arrays = load_posterior(tmp_path / "second", case, second_candidate)
            np.testing.assert_array_equal(first_arrays["parameters"], second_arrays["parameters"])
            np.testing.assert_array_equal(
                first_arrays["predictions_pa"], second_arrays["predictions_pa"]
            )


def test_checkpoints_replay_every_update_and_preserve_member_response_mapping(
    small_campaign: tuple[PressureCase, Path, Path],
) -> None:
    case, snapshot, run = small_campaign
    evidence = snapshot / "evidence"
    selection = read_json(run / "selection.json")
    record = next(
        record
        for record in selection["candidates"]
        if record["family"] == Family.VERTICAL_BAND.value
    )
    hypothesis = parse_hypothesis(record["hypothesis"], case)
    _, manifest = load_case(evidence)
    observations = load_observations(evidence, case, manifest, Phase.CONDITIONING)
    candidate = run / record["candidate_id"]
    settings = read_json(candidate / "candidate.json")
    update_rng = np.random.default_rng(np.random.SeedSequence(settings["update_seed"]))
    smoother = create_smoother(observations["values_pa"], observations["variance_pa2"], update_rng)
    parameters = sample_prior(
        Family.VERTICAL_BAND,
        8,
        np.random.default_rng(np.random.SeedSequence(settings["prior_seed"])),
    )
    model = PressureModel(case)
    for step in range(smoother.num_assimilations()):
        pending_record = read_json(candidate / f"step_{step:02d}_pending.json")
        committed_record = read_json(candidate / f"step_{step:02d}_committed.json")
        pending = read_arrays(
            candidate / f"step_{step:02d}_pending.npz", pending_record["artifact_sha256"]
        )
        committed = read_arrays(
            candidate / f"step_{step:02d}_committed.npz", committed_record["artifact_sha256"]
        )
        assert pending_record["state"] == "update_pending"
        assert committed_record["state"] == "update_committed"
        assert pending_record["step"] == committed_record["step"] == step
        assert pending_record["alpha"] == smoother.alpha[step]
        assert committed_record["pending_sha256"] == pending_record["artifact_sha256"]
        np.testing.assert_array_equal(pending["parameters"], parameters)
        np.testing.assert_array_equal(pending["variance_pa2"], observations["variance_pa2"])
        np.testing.assert_array_equal(pending["observation_ids"], observations["observation_ids"])
        assert pending["member_ids"].tolist() == [f"member_{member:04d}" for member in range(8)]
        np.testing.assert_array_equal(committed["member_ids"], pending["member_ids"])
        np.testing.assert_array_equal(
            pending["parameter_names"], parameter_names(Family.VERTICAL_BAND)
        )
        np.testing.assert_array_equal(committed["parameter_names"], pending["parameter_names"])
        assert update_rng.bit_generator.state == pending_record["rng_state_before"]
        # Exercise restoring serialized RNG state, not just replaying the original seed.
        update_rng.bit_generator.state = pending_record["rng_state_before"]
        perturbations = update_rng.normal(
            0, case.observation_std_pa, size=pending["responses_pa"].shape
        )
        np.testing.assert_array_equal(perturbations, pending["perturbations_pa"])
        assert update_rng.bit_generator.state == pending_record["rng_state_after"]
        for member, values in enumerate(parameters.T):
            field = permeability_field(hypothesis.family, values, case, hypothesis.geometry)
            response = model.simulate(field, 4 * case.segment_steps).responses_pa
            np.testing.assert_array_equal(response, pending["responses_pa"][:, member])
        parameters = update_parameters(smoother, parameters, pending["responses_pa"], perturbations)
        np.testing.assert_array_equal(parameters, committed["parameters"])

    posterior = load_posterior(run, case, record)
    np.testing.assert_array_equal(posterior["parameters"], parameters)
    for member, values in enumerate(parameters.T):
        field = permeability_field(hypothesis.family, values, case, hypothesis.geometry)
        forecast = model.simulate(field, case.total_steps).responses_pa
        np.testing.assert_array_equal(forecast, posterior["responses_pa"][:, member])


def test_over_budget_requests_fail_before_execution(
    tmp_path: Path, small_snapshot: tuple[PressureCase, Path]
) -> None:
    _, snapshot = small_snapshot
    with Campaign(
        snapshot / "evidence", tmp_path / "campaign", members=8, max_evaluations=1
    ) as campaign:
        campaign.execute(campaign.snapshot()["evaluations"][0]["run_id"])
        before = campaign.snapshot()
        assert before["remaining_evaluations"] == before["remaining_simulator_calls"] == 0
        with pytest.raises(ValueError, match="budget"):
            campaign.submit(
                band(0.25),
                "Reject an additional hypothesis before execution",
            )
        assert campaign.snapshot() == before


def test_failed_candidates_are_recorded_without_posteriors(
    tmp_path: Path, small_snapshot: tuple[PressureCase, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, snapshot = small_snapshot

    def fail_member(_model: PressureModel, _field: np.ndarray, _steps: int) -> None:
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(PressureModel, "simulate", fail_member)
    run = tmp_path / "campaign"
    with Campaign(snapshot / "evidence", run, members=8, max_evaluations=2) as campaign:
        campaign.execute(campaign.snapshot()["evaluations"][0]["run_id"])
        proposed = campaign.submit(
            band(0.25),
            "Record a failed revised hypothesis as well as a failed initial hypothesis",
        )
        campaign.execute(proposed["run_id"])
        campaign.finish("uncertain", "No candidate completed the numerical procedure")
        assert campaign.snapshot()["charged_simulator_calls"] == 2
        assert campaign.snapshot()["reserved_simulator_calls"] == 0
    selection = read_json(run / "selection.json")
    assert selection["status"] == "failed"
    assert selection["simulator_calls"] == 2
    for candidate in selection["candidates"]:
        assert candidate["status"] == "failed"
        assert candidate["member_id"] == "member_0000"
        assert "injected forward failure" in candidate["error"]
        assert not (run / candidate["candidate_id"] / "posterior.npz").exists()
    with pytest.raises(ValueError, match="without successful candidates"):
        evaluate_selection(
            snapshot / "evidence", snapshot / "protected", run, tmp_path / "evaluation"
        )


def test_misaligned_observation_ids_are_rejected(
    tmp_path: Path, small_snapshot: tuple[PressureCase, Path]
) -> None:
    _, snapshot = small_snapshot
    evidence = snapshot / "evidence"
    manifest = read_json(evidence / "evidence.json")
    arrays = read_arrays(evidence / "conditioning.npz", manifest["conditioning_sha256"])
    arrays["observation_ids"] = arrays["observation_ids"][::-1]
    np.savez_compressed(evidence / "conditioning.npz", **arrays)
    manifest["conditioning_sha256"] = checksum(evidence / "conditioning.npz")
    (evidence / "evidence.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="IDs/order"):
        Campaign(evidence, tmp_path / "campaign", members=8)
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize("candidate_id", ["vertical_band", "../outside", "h_" + "a" * 25])
def test_posterior_rejects_invalid_candidate_directory_identifiers(
    tmp_path: Path, candidate_id: str
) -> None:
    with pytest.raises(ValueError, match="result-directory identifier"):
        load_posterior(
            tmp_path, PressureCase(), {"family": "vertical_band", "candidate_id": candidate_id}
        )


def test_evaluator_rejects_another_truth_snapshot_and_changed_selection(
    tmp_path: Path, small_campaign: tuple[PressureCase, Path, Path]
) -> None:
    case, snapshot, run = small_campaign
    other = tmp_path / "other_snapshot"
    prepare_case(
        other,
        case,
        Family.HORIZONTAL_BAND,
        noise_seed=102,
        truth_geometry=BandGeometry(0.5, 0.25),
    )
    with pytest.raises(ValueError, match="same measurement files and settings"):
        evaluate_selection(snapshot / "evidence", other / "protected", run, tmp_path / "evaluation")
    selection = read_json(run / "selection.json")
    selection["selected_family"] = "homogeneous"
    (run / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="selection has changed"):
        evaluate_selection(
            snapshot / "evidence", snapshot / "protected", run, tmp_path / "evaluation"
        )
