"""Check that the launcher distinguishes live model requests from prewritten-response tests."""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cases.pressure import PressureCase
from controller.agent_run import run_agent
from controller.artifacts import read_json
from controller.config import RunConfig
from controller.experiment import evaluate_selection, load_posterior, prepare_case
from numerical.geology import BandGeometry, Family


@pytest.fixture
def pi_dependencies() -> None:
    node = shutil.which("node")
    project = Path(__file__).resolve().parents[1]
    if not node or not (project / "agent/node_modules/@earendil-works/pi-agent-core").exists():
        pytest.skip("Pi integration requires the agent dependency setup")
    version = subprocess.run([node, "--version"], check=True, capture_output=True, text=True).stdout
    major, minor, *_patch = map(int, version.strip().lstrip("v").split("."))
    if major < 22 or (major == 22 and minor < 19):
        pytest.skip("Pi integration requires Node >=22.19")


def test_live_mode_never_falls_back_to_scripted_proposals(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    with pytest.raises(ValueError, match="explicit Copilot model"):
        run_agent(tmp_path / "evidence", output, RunConfig(output=output, model=None))
    with pytest.raises(ValueError, match="cannot use a live"):
        run_agent(
            tmp_path / "evidence",
            output,
            RunConfig(output=output, model="model"),
            faux=True,
        )
    assert not output.exists()


@pytest.mark.parametrize("version", ["v20.19.0\n", "v22.18.0\n"])
def test_node_executable_uses_path_and_rejects_older_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    node_lookup = Mock(return_value="/path/node")
    monkeypatch.setattr("controller.agent_run.shutil.which", node_lookup)
    version_check = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=version))
    monkeypatch.setattr("controller.agent_run.subprocess.run", version_check)
    output = tmp_path / "campaign"
    with pytest.raises(ValueError, match="Node >=22.19 is required on PATH"):
        run_agent(
            tmp_path / "evidence",
            output,
            RunConfig(output=output, model=None),
            faux=True,
        )
    node_lookup.assert_called_once_with("node")
    version_check.assert_called_once_with(
        ["/path/node", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not output.exists()


@pytest.mark.parametrize("max_evaluations", [2, 3])
@pytest.mark.usefixtures("pi_dependencies")
def test_operator_launcher_runs_the_scripted_bridge_without_disclosing_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_evaluations: int,
) -> None:
    project = Path(__file__).resolve().parents[1]
    case = PressureCase(grid_size=8, segment_steps=4, sample_every=1)
    snapshot = tmp_path / "snapshot"
    prepare_case(
        snapshot,
        case,
        truth_geometry=BandGeometry(0.5, 0.25),
    )
    output = tmp_path / "campaign"
    config = RunConfig(
        output=snapshot,
        model=None,
        realizations=16,
        max_evaluations=max_evaluations,
    )
    node_process = Mock(wraps=subprocess.run)
    monkeypatch.setattr("controller.agent_run.subprocess.run", node_process)
    state = run_agent(snapshot / "evidence", output, config, faux=True)
    commands = [
        call.args[0]
        for call in node_process.call_args_list
        if call.kwargs.get("cwd") == project / "agent"
    ]
    assert len(commands) == 1
    assert commands[0][:3] == [
        shutil.which("node"),
        "--use-system-ca",
        "--experimental-strip-types",
    ]
    assert state["closed"] and state["outcome"] == "adequate"
    assert state["proposer_mode"] == "faux_scripted_smoke_test"
    assert state["model_calls"] == 2
    assert state["charged_simulator_calls"] == 160
    assert state["reserved_simulator_calls"] == 0
    assert state["remaining_simulator_calls"] == 80 * (max_evaluations - 2)
    assert json.loads((output / "campaign.json").read_text())["max_simulations"] == (
        80 * max_evaluations
    )
    assert "truth_family" not in (output / "proposer" / "messages.jsonl").read_text()
    report = evaluate_selection(
        snapshot / "evidence", snapshot / "protected", output, tmp_path / "grade"
    )
    assert report["structural_truth_recovered"] is True
    assert report["proposer_mode"] == "faux_scripted_smoke_test"


@pytest.mark.usefixtures("pi_dependencies")
def test_default_scripted_experiment_fits_and_grades_the_fixed_case(tmp_path: Path) -> None:
    """Run the README's fixed case with 50 realizations and grade its frozen selection."""
    case = PressureCase()
    snapshot = tmp_path / "snapshot"
    prepare_case(
        snapshot,
        case,
        truth_geometry=BandGeometry(0.5, 0.1),
    )
    evidence = snapshot / "evidence"
    campaign = tmp_path / "campaign"
    state = run_agent(evidence, campaign, RunConfig(output=snapshot, model=None), faux=True)
    assert state["closed"] and state["outcome"] == "adequate"
    assert state["model_calls"] == 2
    assert state["charged_simulator_calls"] == 500
    assert state["reserved_simulator_calls"] == 0

    selection = read_json(campaign / "selection.json")
    proposed = next(
        record
        for record in selection["candidates"]
        if record["candidate_id"] == selection["selected_candidate_id"]
    )
    assert proposed["family"] == Family.VERTICAL_BAND.value
    fitted = load_posterior(campaign, case, proposed)
    assert fitted["parameters"].shape == (2, 50)
    assert fitted["responses_pa"].shape == (128, 50)
    assert fitted["predictions_pa"].shape == (128, 50)

    report = evaluate_selection(evidence, snapshot / "protected", campaign, tmp_path / "grade")
    assert report["structural_truth_recovered"] is True
    assert report["parameter_recovery"]["background_md"]["relative_median_error"] < 0.15
    assert report["parameter_recovery"]["band_ratio"]["relative_median_error"] < 0.35
    assert report["proposer_mode"] == "faux_scripted_smoke_test"
    assert report["model_cost_estimate_usd"] == 0.0
