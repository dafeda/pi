"""Check data generation, live model search, hidden-data grading, and stopping on errors."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pytest
import yaml

from cases.pressure import PressureCase
from controller import __main__ as cli
from controller.agent_run import run_agent
from controller.config import RunConfig
from numerical.geology import MILLIDARCY_M2, BandGeometry


@pytest.fixture
def operations(monkeypatch: pytest.MonkeyPatch) -> Mock:
    operations = Mock()
    operations.prepare_case.return_value = {"evidence": "public", "protected": "operator-only"}
    operations.run_agent.return_value = {"closed": True, "proposer_mode": "live_llm"}
    operations.evaluate_selection.return_value = {
        "selected_family": "vertical_band",
        "truth_family": "vertical_band",
        "family_recovered": True,
        "parameter_recovery": {},
        "proposer_mode": "live_llm",
        "model_calls": 2,
        "simulator_calls": 120,
        "wall_seconds": 0.5,
    }
    for name in ("prepare_case", "run_agent", "evaluate_selection"):
        monkeypatch.setattr(cli, name, getattr(operations, name))
    return operations


def test_one_config_prepares_runs_a_live_agent_and_grades_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operations: Mock,
) -> None:
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "run.yaml"
    path.write_text(
        "output: ../results\nmodel: fixture-model\n"
        "realizations: 8\nseed: 11\nmax_evaluations: 2\n"
        "max_model_calls: 4\nmax_model_cost_usd: 1.25\n"
    )
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    cli.main()
    captured = capsys.readouterr()
    assert "Run complete in" in captured.err
    output = tmp_path / "results"
    case = PressureCase()
    assert operations.mock_calls == [
        call.prepare_case(
            output,
            case,
            truth_geometry=BandGeometry(0.5, 0.1),
        ),
        call.run_agent(
            output / "evidence",
            output / "search",
            RunConfig(
                output=output,
                model="fixture-model",
                realizations=8,
                seed=11,
                max_evaluations=2,
                max_model_calls=4,
                max_model_cost_usd=1.25,
            ),
        ),
        call.evaluate_selection(
            output / "evidence",
            output / "protected",
            output / "search",
            output / "protected/evaluation",
        ),
    ]
    assert json.loads(captured.out) == {
        **operations.evaluate_selection.return_value,
        "report": str(output / "protected/evaluation/report.json"),
    }


def test_minimal_config_uses_live_mode_and_the_fixed_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operations: Mock,
) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("output: results\nmodel: fixture-model\n")
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    cli.main()
    case = PressureCase()
    operations.prepare_case.assert_called_once_with(
        tmp_path / "results",
        case,
        truth_geometry=BandGeometry(0.5, 0.1),
    )
    operations.run_agent.assert_called_once_with(
        tmp_path / "results/evidence",
        tmp_path / "results/search",
        RunConfig(output=tmp_path / "results", model="fixture-model"),
    )


def test_inference_settings_leave_the_fixed_truth_and_observations_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = Mock(return_value={"closed": True})
    monkeypatch.setattr(cli, "run_agent", launcher)
    monkeypatch.setattr(cli, "evaluate_selection", Mock(return_value={}))
    outputs: list[Path] = []
    for seed, realizations in ((7, 8), (11, 16)):
        output = tmp_path / f"run-{seed}"
        path = tmp_path / f"run-{seed}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "output": output.name,
                    "model": "fixture-model",
                    "seed": seed,
                    "realizations": realizations,
                }
            )
        )
        monkeypatch.setattr(sys, "argv", ["controller", str(path)])
        cli.main()
        outputs.append(output)
    assert [
        (record.args[2].seed, record.args[2].realizations) for record in launcher.call_args_list
    ] == [(7, 8), (11, 16)]
    expected_truth = {
        "family": "vertical_band",
        "background_md": 100.0,
        "band_ratio": 0.01,
        "geometry": {"center_fraction": 0.5, "width_fraction": 0.1},
        "noise_seed": 2026,
    }
    for output in outputs:
        truth = json.loads((output / "protected/truth.json").read_text())
        assert {name: truth[name] for name in expected_truth} == expected_truth
        public = (output / "evidence/evidence.json").read_text()
        assert "band_width_fraction" not in public
        assert "vertical_band" not in public
        assert "noise_seed" not in public
    first, second = outputs
    for relative in (
        "protected/truth.npz",
        "evidence/conditioning.npz",
        "evidence/validation.npz",
        "protected/final.npz",
    ):
        with np.load(first / relative) as first_arrays, np.load(second / relative) as second_arrays:
            assert first_arrays.files == second_arrays.files
            for name in first_arrays.files:
                np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
            if relative == "protected/truth.npz":
                expected_md = np.full((20, 20), 100.0)
                expected_md[:, 9:11] = 1.0
                np.testing.assert_allclose(
                    first_arrays["permeability_m2"] / MILLIDARCY_M2, expected_md, rtol=1e-12
                )


def test_existing_output_is_cleared_before_preparing_a_fresh_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operations: Mock,
) -> None:
    output = tmp_path / "results"
    ledger = output / "search/ledger.sqlite"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("old campaign state")
    outside = tmp_path / "keep.txt"
    outside.write_text("unrelated operator data")

    def prepare_fresh_run(path: Path, case: PressureCase, *, truth_geometry: BandGeometry) -> None:
        assert path == output
        assert case == PressureCase()
        assert truth_geometry == BandGeometry(0.5, 0.1)
        assert not path.exists()
        path.mkdir()
        (path / "fresh.txt").write_text("new run")

    operations.prepare_case.side_effect = prepare_fresh_run
    path = tmp_path / "run.yaml"
    path.write_text("output: results\nmodel: fixture-model\n")
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    cli.main()
    case = PressureCase()
    operations.prepare_case.assert_called_once_with(
        output,
        case,
        truth_geometry=BandGeometry(0.5, 0.1),
    )
    operations.run_agent.assert_called_once()
    operations.evaluate_selection.assert_called_once()
    assert not ledger.exists()
    assert (output / "fresh.txt").read_text() == "new run"
    assert outside.read_text() == "unrelated operator data"
    captured = capsys.readouterr()
    assert "Run complete in" in captured.err
    assert json.loads(captured.out)["report"] == str(output / "protected/evaluation/report.json")


@pytest.mark.parametrize("target", ["config", "working_directory", "controller", "ancestor"])
def test_output_cannot_contain_the_config_working_directory_or_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operations: Mock,
    target: str,
) -> None:
    directory = tmp_path / "configs"
    directory.mkdir()
    working_directory = tmp_path / "workspace"
    working_directory.mkdir()
    controller = tmp_path / "controller/__main__.py"
    controller.parent.mkdir()
    controller.write_text("# controller source\n")
    output = {
        "config": directory,
        "working_directory": working_directory,
        "controller": controller.parent,
        "ancestor": tmp_path,
    }[target]
    marker = output / "keep.txt"
    marker.write_text("existing operator data")
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump({"output": str(output), "model": "fixture-model"}))
    monkeypatch.chdir(working_directory)
    monkeypatch.setattr(cli, "__file__", str(controller))
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert operations.mock_calls == []
    assert marker.read_text() == "existing operator data"
    assert path.exists() and controller.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "output must not contain" in captured.err


def test_invalid_config_does_not_remove_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operations: Mock,
) -> None:
    output = tmp_path / "results"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing operator data")
    path = tmp_path / "run.yaml"
    path.write_text("output: results\nmodel: fixture-model\nmax_evaluations: 0\n")
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert operations.mock_calls == []
    assert marker.read_text() == "existing operator data"


def test_output_cleanup_failure_stops_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operations: Mock,
) -> None:
    output = tmp_path / "results"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing operator data")
    remove = Mock(side_effect=PermissionError("cannot remove old output"))
    monkeypatch.setattr(cli.shutil, "rmtree", remove)
    path = tmp_path / "run.yaml"
    path.write_text("output: results\nmodel: fixture-model\n")
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    remove.assert_called_once_with(output)
    assert operations.mock_calls == []
    assert marker.read_text() == "existing operator data"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot remove old output" in captured.err


@pytest.mark.parametrize(
    "content",
    [
        "output: results\n",
        "output: results\nmodel: fixture-model\nmax_simulations: 1\n",
        "output: results\nmodel: fixture-model\nmembers: 50\n",
        "output: results\nmodel: fixture-model\ntruth_center: 0.65\n",
        "output: results\nmodel: fixture-model\nfaux: true\n",
        "output: results\nmodel: fixture-model\ncommand: demo\n",
        "output: results\nmodel: fixture-model\nprovider: openai\n",
        "output: results\nmodel: fixture-model\nnode: /opt/node\n",
        "output: [\n",
    ],
)
def test_invalid_configs_exit_cleanly_before_any_scientific_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operations: Mock,
    content: str,
) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(content)
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert operations.mock_calls == []
    assert not (tmp_path / "results").exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(path) in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "failure", [ValueError("bad case"), OSError("bad file"), RuntimeError("failed")]
)
@pytest.mark.parametrize("phase", ["prepare_case", "run_agent", "evaluate_selection"])
def test_phase_errors_stop_later_work_and_exit_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operations: Mock,
    failure: Exception,
    phase: str,
) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("output: results\nmodel: fixture-model\n")
    getattr(operations, phase).side_effect = failure
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    phases = ["prepare_case", "run_agent", "evaluate_selection"]
    for index, name in enumerate(phases):
        assert getattr(operations, name).call_count == (1 if index <= phases.index(phase) else 0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(failure) in captured.err


def test_failed_search_retains_prepared_evidence_and_protected_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = PressureCase(segment_steps=4, sample_every=1)
    monkeypatch.setattr(cli, "PressureCase", Mock(return_value=case))
    monkeypatch.setattr(cli, "run_agent", Mock(side_effect=RuntimeError("proposer failed")))
    evaluate = Mock()
    monkeypatch.setattr(cli, "evaluate_selection", evaluate)
    path = tmp_path / "run.yaml"
    path.write_text("output: results\nmodel: fixture-model\n")
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    evaluate.assert_not_called()
    assert (tmp_path / "results/evidence/evidence.json").exists()
    assert (tmp_path / "results/protected/truth.json").exists()


def test_missing_yaml_file_exits_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "missing.yaml"
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(path) in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("arguments", "exit_code"),
    [
        ([], 2),
        (["--help"], 0),
        (["run.yaml", "another.yaml"], 2),
        (["agent", "run.yaml"], 2),
        (["run.yaml", "--faux"], 2),
    ],
)
def test_cli_accepts_only_one_config_path_or_help(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], exit_code: int
) -> None:
    monkeypatch.setattr(sys, "argv", ["controller", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == exit_code


def test_help_explains_the_model_role_hidden_data_and_copilot_quota(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], operations: Mock
) -> None:
    monkeypatch.setattr(sys, "argv", ["controller", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "live LLM" in help_text
    assert "GitHub Copilot" in help_text
    assert "YAML config" in help_text
    assert "LLM proposes model structures; ES-MDA fits permeability parameters" in help_text
    assert "Truth and final observations are withheld during search" in help_text
    assert "Each run deletes existing output" in help_text
    assert "requires output and model" in help_text
    assert "live requests use Copilot quota" in help_text
    assert operations.mock_calls == []


def test_complete_workflow_with_an_internal_scripted_launcher_and_real_numerics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    node = shutil.which("node")
    project = Path(__file__).resolve().parents[1]
    if not node or not (project / "agent/node_modules/@earendil-works/pi-agent-core").exists():
        pytest.skip("Pi integration requires Node >=22.19 and the agent dependency setup")
    version = subprocess.run([node, "--version"], check=True, capture_output=True, text=True).stdout
    major, minor, *_patch = map(int, version.strip().lstrip("v").split("."))
    if major < 22 or (major == 22 and minor < 19):
        pytest.skip("Pi integration requires Node >=22.19")
    case = PressureCase(segment_steps=4, sample_every=1)
    monkeypatch.setattr(cli, "PressureCase", Mock(return_value=case))

    def scripted_agent(
        evidence: Path,
        output: Path,
        config: RunConfig,
    ) -> dict[str, object]:
        assert config.model == "fixture-model"
        return run_agent(
            evidence,
            output,
            replace(config, model=None),
            faux=True,
        )

    monkeypatch.setattr(cli, "run_agent", scripted_agent)
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "output": "../results",
                "model": "fixture-model",
                "realizations": 16,
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["controller", str(path)])
    output = tmp_path / "results"
    report_path = output / "protected/evaluation/report.json"
    for _ in range(2):
        cli.main()
        captured = capsys.readouterr()
        assert "Run complete in" in captured.err
        result = json.loads(captured.out)
        assert result["report"] == str(report_path)
        assert result["simulator_calls"] == 160
        assert result["model_calls"] == 2
        assert result["proposer_mode"] == "faux_scripted_smoke_test"
        assert result["structural_truth_recovered"] is True
        report = json.loads(report_path.read_text())
        assert report["selected_family"] == result["selected_family"]
        assert report["truth_simulator_calls"] == 1
        assert (output / "evidence/evidence.json").exists()
        assert (output / "search/selection.json").exists()
        transcript = (output / "search/proposer/messages.jsonl").read_text()
        assert "truth_family" not in transcript
        assert "noise_seed" not in transcript
        assert str(output / "protected") not in transcript
        assert str(path) not in transcript
        assert (output / "protected/evaluation/diagnostics.png").read_bytes().startswith(b"\x89PNG")
