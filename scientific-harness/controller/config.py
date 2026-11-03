"""Load an operator-only YAML config for the fixed-truth LLM experiment."""

from dataclasses import MISSING, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import get_args

import yaml
from yaml.nodes import MappingNode


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """Configure the model, inference settings, and budgets without changing truth or observations.

    YAML requires a Copilot model; scripted tests use None.
    max_model_cost_usd caps catalog-estimated cost, not Copilot quota.
    """

    output: Path
    model: str | None
    realizations: int = 50
    seed: int = 7
    max_evaluations: int = 3
    max_model_calls: int = 6
    max_model_cost_usd: float = 5.0


def load_config(path: Path) -> RunConfig:
    """Validate YAML settings and resolve output relative to the config file."""
    with path.open(encoding="utf-8") as stream:
        try:
            value: object = yaml.load(stream, Loader=_ConfigLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping with string keys")
    config_fields = fields(RunConfig)
    unknown = set(value) - {field.name for field in config_fields}
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")
    missing = [
        field.name
        for field in config_fields
        if field.default is MISSING and field.name not in value
    ]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
    directory = path.resolve().parent
    config = RunConfig(
        **{
            field.name: _parse_value(
                field.name, value.get(field.name, field.default), field.type, directory
            )
            for field in config_fields
        }
    )
    _validate_config(config)
    return config


class _ConfigLoader(yaml.SafeLoader):
    """Use safe YAML loading while rejecting duplicate config keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[str, object]:
        if not isinstance(node, MappingNode):
            raise ValueError("config must be a mapping with string keys")
        keys: set[str] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("config keys must be strings")
            if key in keys:
                raise ValueError(f"duplicate config key: {key}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _parse_value(name: str, value: object, annotation: object, directory: Path) -> object:
    nullable = get_args(annotation)
    if nullable:
        if value is None:
            return None
        annotation = nullable[0]
    if annotation in (str, Path):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if annotation is Path:
            return (directory / value).resolve()
        return value
    if annotation is float:
        try:
            if type(value) in (int, float) and isfinite(value):
                return float(value)
        except OverflowError:
            # YAML integers can exceed floating-point range.
            pass
        raise ValueError(f"{name} must be a finite number")
    if type(value) is not annotation:
        raise ValueError(f"{name} must be {annotation.__name__}")
    return value


def _validate_config(config: RunConfig) -> None:
    if config.model is None:
        raise ValueError("model must be a non-empty string")
    if config.realizations < 2:
        raise ValueError("realizations must be an integer of at least two")
    if config.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if config.max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if config.max_model_calls < 1:
        raise ValueError("max_model_calls must be a positive integer")
    if config.max_model_cost_usd <= 0:
        raise ValueError("max_model_cost_usd must be positive")
