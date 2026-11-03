"""Launch the Pi host against the loopback controller."""

import json
import shutil
import subprocess
import threading
from pathlib import Path

from controller.campaign import Campaign
from controller.config import RunConfig
from controller.progress import logger
from controller.server import ScientificServer


def run_agent(
    evidence: Path,
    output: Path,
    config: RunConfig,
    *,
    faux: bool = False,
) -> dict[str, object]:
    """Launch trusted host code; the proposer receives only public evidence and reviewed tools."""
    if faux and config.model:
        raise ValueError("scripted test mode cannot use a live model")
    if not faux and not config.model:
        raise ValueError("live mode requires an explicit Copilot model")
    logger.info("Checking Pi host dependencies")
    executable = shutil.which("node")
    if not executable:
        raise ValueError("Node >=22.19 is required on PATH for the Pi proposer")
    version = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True
    ).stdout
    major, minor, *_patch = map(int, version.strip().lstrip("v").split("."))
    if major < 22 or (major == 22 and minor < 19):
        raise ValueError("Node >=22.19 is required on PATH for the Pi proposer")
    agent = Path(__file__).resolve().parents[1] / "agent"
    if not (agent / "node_modules" / "@earendil-works" / "pi-agent-core").exists():
        raise ValueError("install the compatible local Pi packages in agent/ first; see README.md")
    logger.info("Pi host dependencies ready: Node %s", version.strip())
    with Campaign(
        evidence,
        output,
        members=config.realizations,
        seed=config.seed,
        max_evaluations=config.max_evaluations,
        max_model_calls=config.max_model_calls,
        max_model_cost_usd=config.max_model_cost_usd,
    ) as campaign:
        if campaign.snapshot()["closed"]:
            logger.info("Search already frozen; no proposer launch needed")
            return campaign.snapshot()
        logger.info("Starting loopback controller and numerical worker")
        with ScientificServer(campaign, port=0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            # Node's bundled roots may omit certificate authorities trusted by the OS.
            command = [
                executable,
                "--use-system-ca",
                "--experimental-strip-types",
                "src/cli.ts",
                "--url",
                f"http://127.0.0.1:{server.server_port}/",
                "--audit-dir",
                str(output.resolve() / "proposer"),
                "--max-turns",
                str(config.max_model_calls),
            ]
            command += ["--faux"] if faux else ["--model", config.model]
            if faux:
                logger.info("Launching scripted proposer; no live model requests")
            else:
                logger.info(
                    "Launching Pi proposer; resolving saved Copilot login for %s", config.model
                )
            try:
                process = subprocess.run(command, cwd=agent, capture_output=True, text=True)
            finally:
                server.shutdown()
                thread.join()
        state = campaign.snapshot()
        if process.returncode:
            if not state["closed"]:
                campaign.finish("uncertain", "Proposer process failed; accepted work retained")
            raise RuntimeError(process.stderr.strip() or "proposer process failed")
        json.loads(process.stdout)
        logger.info(
            "Proposer complete: transcript=%s; summary=%s",
            output / "proposer" / "messages.jsonl",
            output / "proposer" / "summary.json",
        )
        return {**state, "proposer_mode": "faux_scripted_smoke_test" if faux else "live_llm"}
