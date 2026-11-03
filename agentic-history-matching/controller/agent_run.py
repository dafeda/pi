"""Run the Python proposer and numerical evaluations sequentially against Campaign."""

from pathlib import Path

from controller.campaign import Campaign
from controller.config import RunConfig
from controller.progress import logger
from controller.proposer import run_proposer


def run_agent(evidence: Path, output: Path, config: RunConfig) -> dict[str, object]:
    """Run bounded search from the synchronous CLI and return the frozen public state.

    The model receives only public evidence/evaluations and two domain tools.
    No Python objects, files, shell, operator settings, or protected data enter its context.
    Numerical evaluations finish before the next model request; failures abort the run.
    """
    campaign = Campaign(
        evidence,
        output,
        realizations=config.realizations,
        seed=config.seed,
        max_evaluations=config.max_evaluations,
        max_model_calls=config.max_model_calls,
    )
    logger.info("Starting sequential numerical evaluations and bounded Python proposer")
    state = run_proposer(campaign, config.model)
    logger.info(
        "Proposer complete: transcript=%s; summary=%s",
        output / "proposer" / "messages.jsonl",
        output / "proposer" / "summary.json",
    )
    return state
