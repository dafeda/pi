"""Check name: value settings, rejected inputs, and output paths relative to the settings file."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from controller.config import RunConfig, load_config


def test_minimal_config_requires_a_live_model_and_preserves_defaults(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("output: results\nmodel: fixture-model\n")
    config = load_config(path)
    assert isinstance(config, RunConfig)
    assert asdict(config) == {
        "output": tmp_path / "results",
        "model": "fixture-model",
        "realizations": 50,
        "seed": 7,
        "max_evaluations": 3,
        "max_model_calls": 6,
        "max_model_cost_usd": 5.0,
    }


def test_explicit_values_are_converted_without_string_coercion(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "output: runs/experiment\nmodel: fixture-model\n"
        "realizations: 8\nseed: 11\n"
        "max_evaluations: 2\nmax_model_calls: 4\nmax_model_cost_usd: 1\n"
    )
    config = load_config(path)
    assert config == RunConfig(
        output=tmp_path / "runs/experiment",
        model="fixture-model",
        realizations=8,
        seed=11,
        max_evaluations=2,
        max_model_calls=4,
        max_model_cost_usd=1.0,
    )
    assert type(config.max_model_cost_usd) is float


@pytest.mark.parametrize("relative", [True, False])
def test_output_is_config_relative_and_absolute_paths_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool
) -> None:
    directory = tmp_path / "configs"
    directory.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = directory / "run.yaml"
    output = "../results" if relative else str(tmp_path / "results")
    path.write_text(f"output: {output}\nmodel: fixture-model\n")
    monkeypatch.chdir(elsewhere)
    assert load_config(path) == RunConfig(output=tmp_path / "results", model="fixture-model")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("max_evaluations", [1, 33, 1000])
def test_search_budget_accepts_positive_evaluation_limits(
    tmp_path: Path, max_evaluations: int
) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(f"output: results\nmodel: fixture-model\nmax_evaluations: {max_evaluations}\n")
    config = load_config(path)
    assert config.max_evaluations == max_evaluations


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("", "mapping"),
        ("null\n", "mapping"),
        ("run\n", "mapping"),
        ("- output: results\n", "mapping"),
        ("{}\n", "missing required fields: model, output"),
        ("output: results\n", "missing required fields: model"),
        ("model: fixture-model\n", "missing required fields: output"),
        ("1: value\n", "keys must be strings"),
        ("? [a, b]\n: value\n", "keys must be strings"),
        ("model: first\nmodel: second\n", "duplicate config key: model"),
        ("seed: 7\nseed: 11\n", "duplicate config key: seed"),
        ("output: [\n", "invalid YAML"),
        ("output: first\n---\noutput: second\n", "invalid YAML"),
        ("<<: {seed: 11}\n", "invalid YAML"),
        ("!!python/object/apply:builtins.len [[1, 2]]\n", "invalid YAML"),
        ("!!map scalar\n", "mapping"),
    ],
)
def test_invalid_yaml_and_missing_fields_fail_without_creating_artifacts(
    tmp_path: Path, content: str, error: str
) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match=error):
        load_config(path)
    assert sorted(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"command": "demo"}, "unsupported fields: command"),
        ({"faux": True}, "unsupported fields: faux"),
        ({"faux": False}, "unsupported fields: faux"),
        ({"evidence": "snapshot"}, "unsupported fields: evidence"),
        ({"protected": "secret"}, "unsupported fields: protected"),
        ({"run": "old-search"}, "unsupported fields: run"),
        ({"strategy": "rule"}, "unsupported fields: strategy"),
        ({"port": 8910}, "unsupported fields: port"),
        ({"truth_family": "vertical_band"}, "unsupported fields: truth_family"),
        ({"background_md": 100.0}, "unsupported fields: background_md"),
        ({"band_ratio": 0.01}, "unsupported fields: band_ratio"),
        ({"noise_seed": 2026}, "unsupported fields: noise_seed"),
        ({"truth_center": 0.5}, "unsupported fields: truth_center"),
        ({"truth_width": 0.1}, "unsupported fields: truth_width"),
        ({"max_simulation": 750}, "unsupported fields: max_simulation"),
        ({"max_simulations": 750}, "unsupported fields: max_simulations"),
        ({"members": 50}, "unsupported fields: members"),
        ({"output": None}, "output must be a non-empty string"),
        ({"output": ""}, "output must be a non-empty string"),
        ({"output": True}, "output must be a non-empty string"),
        ({"output": ["run"]}, "output must be a non-empty string"),
        ({"provider": "github-copilot"}, "unsupported fields: provider"),
        ({"provider": "openai"}, "unsupported fields: provider"),
        ({"model": None}, "model must be a non-empty string"),
        ({"model": ""}, "model must be a non-empty string"),
        ({"model": False}, "model must be a non-empty string"),
        ({"realizations": "8"}, "realizations must be int"),
        ({"realizations": 8.0}, "realizations must be int"),
        ({"realizations": True}, "realizations must be int"),
        ({"realizations": None}, "realizations must be int"),
        ({"realizations": 1}, "realizations must be an integer of at least two"),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"max_evaluations": 0}, "max_evaluations must be a positive integer"),
        ({"max_evaluations": -1}, "max_evaluations must be a positive integer"),
        ({"max_model_calls": 0}, "max_model_calls must be a positive integer"),
        ({"max_model_cost_usd": "1"}, "max_model_cost_usd must be a finite number"),
        ({"max_model_cost_usd": True}, "max_model_cost_usd must be a finite number"),
        ({"max_model_cost_usd": float("nan")}, "max_model_cost_usd must be a finite number"),
        ({"max_model_cost_usd": 0}, "max_model_cost_usd must be positive"),
        ({"max_model_cost_usd": float("inf")}, "max_model_cost_usd must be a finite number"),
        ({"max_model_cost_usd": 10**400}, "max_model_cost_usd must be a finite number"),
        ({"node": "/opt/node"}, "unsupported fields: node"),
        ({"node": None}, "unsupported fields: node"),
        ({"node": ""}, "unsupported fields: node"),
    ],
)
def test_invalid_settings_fail_before_creating_artifacts(
    tmp_path: Path, settings: dict[str, object], error: str
) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "output": "results",
                "model": "fixture-model",
                **settings,
            }
        )
    )
    with pytest.raises(ValueError, match=error):
        load_config(path)
    assert sorted(tmp_path.iterdir()) == [path]


def test_missing_config_reports_a_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_single_shipped_example_is_valid_without_running_it() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/run.yaml"
    config = load_config(path)
    assert isinstance(config, RunConfig)
    assert config.output.is_absolute()
    assert config.realizations == 50
