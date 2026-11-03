"""Run preparation, live LLM search, and protected evaluation from one YAML config."""

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

from cases.pressure import PressureCase
from controller.agent_run import run_agent
from controller.config import load_config
from controller.experiment import evaluate_selection, prepare_case
from controller.progress import logger, terminal_progress
from numerical.geology import BandGeometry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer flow barriers from synthetic pressure observations "
            "with a live LLM on GitHub Copilot"
        ),
        epilog=(
            "LLM proposes model structures; ES-MDA fits permeability parameters. "
            "Truth and final observations are withheld during search. "
            "Each run deletes existing output; live requests use Copilot quota. "
            "Progress is written to stderr; the final JSON report is written to stdout."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="YAML config; requires output and model",
    )
    args = parser.parse_args()

    with terminal_progress():
        started = perf_counter()
        stage = "configuration"
        output: Path | None = None
        try:
            logger.info("Loading config: %s", args.config)
            config = load_config(args.config)
            stage = "output validation"
            if any(
                path.is_relative_to(config.output)
                for path in (args.config.resolve(), Path.cwd(), Path(__file__).resolve())
            ):
                raise ValueError(
                    "output must not contain the config, working directory, or controller source"
                )
            output = config.output
            logger.info(
                "Run started: model=%s; realizations=%d; inference seed=%d; output=%s",
                config.model,
                config.realizations,
                config.seed,
                output,
            )
            stage = "output cleanup"
            if output.exists():
                logger.info("Clearing previous output: %s", output)
                shutil.rmtree(output)
                logger.info("Previous output cleared")
            stage = "case preparation"
            logger.info("Preparing synthetic observations; truth and final readings stay protected")
            case = PressureCase()
            prepare_case(
                output,
                case,
                truth_geometry=BandGeometry(0.5, 0.1),
            )
            evidence = output / "evidence"
            protected = output / "protected"
            search = output / "search"
            logger.info("Preparation complete: evidence=%s; protected=%s", evidence, protected)
            stage = "search"
            logger.info("Search started: fit the homogeneous baseline, then request LLM proposals")
            run_agent(evidence, search, config)
            logger.info("Search complete: frozen selection=%s", search / "selection.json")
            stage = "protected evaluation"
            logger.info("Protected evaluation started: verify the frozen selection before grading")
            evaluation = protected / "evaluation"
            report = evaluate_selection(evidence, protected, search, evaluation)
            logger.info("Protected evaluation complete")
            stage = "report output"
            print(
                json.dumps(
                    {**report, "report": str(evaluation / "report.json")}, indent=2, allow_nan=False
                )
            )
            logger.info(
                "Run complete in %.2fs: report=%s; diagnostics=%s",
                perf_counter() - started,
                evaluation / "report.json",
                evaluation / "diagnostics.png",
            )
        except (ValueError, OSError, RuntimeError) as error:
            logger.error("Run failed during %s (%s): %s", stage, args.config, error)
            if output is not None and output.exists():
                logger.info(
                    "Output retained for inspection: %s; the next invocation clears it", output
                )
            parser.exit(1)


if __name__ == "__main__":
    main()
