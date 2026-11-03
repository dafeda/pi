"""Own queued evaluations, transactional budget reservations, and frozen search decisions."""

import fcntl
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from threading import Lock

from cases.pressure import Phase
from controller.artifacts import checksum, read_json, write_json
from controller.assimilation import INFLATION
from controller.candidate import condition_candidate
from controller.experiment import implementation, load_case, load_observations
from controller.hypotheses import parse_hypothesis
from controller.local import LocalExecutor
from controller.progress import logger
from numerical.pressure import PressureModel

# Bound local numerical work independently of operator-selected search depth.
MAX_GRID_SIZE = 64
MAX_TIME_STEPS = 512
MAX_CONDITIONING_TIME_SAMPLES = 128

MAX_REASON_LENGTH = 2000
MAX_MODEL_NAME_LENGTH = 200

# Allow floating-point rounding when summing catalog-estimated USD costs.
MODEL_COST_ROUNDING_TOLERANCE_USD = 1e-10


class Campaign:
    """Run one hypothesis search under fixed conditioning policy and budgets.

    Conditioning observations fit parameters; validation observations rank candidates.
    ledger.sqlite persists evaluations and budget accounting across controller restarts.
    """

    def __init__(
        self,
        evidence: Path,
        output: Path,
        members: int = 50,
        seed: int = 7,
        max_evaluations: int = 3,
        max_model_calls: int = 6,
        max_model_cost_usd: float = 5.0,
    ) -> None:
        for name, value, minimum in (
            ("members", members, 2),
            ("seed", seed, 0),
            ("max_evaluations", max_evaluations, 1),
            ("max_model_calls", max_model_calls, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isfinite(max_model_cost_usd) or max_model_cost_usd <= 0:
            raise ValueError("model cost budget must be finite and positive")
        self.case, self.evidence_manifest = load_case(evidence)
        if (
            self.case.grid_size > MAX_GRID_SIZE
            or self.case.total_steps > MAX_TIME_STEPS
            or self.case.phase_mask(Phase.CONDITIONING).sum()
            > MAX_CONDITIONING_TIME_SAMPLES * len(self.case.gauge_ids)
        ):
            raise ValueError(
                "experiment exceeds the laptop controller's grid, time-step, "
                "or pressure-reading limits"
            )
        self.conditioning = load_observations(
            evidence, self.case, self.evidence_manifest, Phase.CONDITIONING
        )
        self.validation = load_observations(
            evidence, self.case, self.evidence_manifest, Phase.VALIDATION
        )
        self.evidence = evidence
        self.output = output
        self.members = members
        self.seed = seed
        self.max_model_calls = max_model_calls
        self.max_model_cost_usd = max_model_cost_usd
        self.planned_calls = (len(INFLATION) + 1) * members
        self.max_simulations = self.planned_calls * max_evaluations
        self.max_evaluations = max_evaluations
        self.model = PressureModel(self.case)
        settings = {
            "schema_version": 1,
            "evidence_sha256": checksum(evidence / "evidence.json"),
            "members": members,
            "seed": seed,
            "max_simulations": self.max_simulations,
            "max_evaluations": max_evaluations,
            "implementation": implementation(),
            "max_model_calls": max_model_calls,
            "max_model_cost_usd": max_model_cost_usd,
            "adequacy_rule": (
                "conditioning and validation RMSE <= 2 observation standard deviations"
            ),
        }
        if output.exists():
            if read_json(output / "campaign.json") != settings:
                raise ValueError(
                    "search settings, source files, or package versions changed; "
                    "use a new output directory"
                )
        else:
            output.mkdir(parents=True)
            write_json(output / "campaign.json", settings)
        self.owner = (output / ".controller.lock").open("a")
        try:
            fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise ValueError("another controller owns this campaign") from None
        self.execution_lock = Lock()
        self.publication_lock = Lock()
        self.database = output / "ledger.sqlite"
        with closing(self.connect()) as connection, connection:
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, "
                "spent INTEGER NOT NULL, "
                "reserved INTEGER NOT NULL, closed INTEGER NOT NULL, outcome TEXT, reason TEXT);"
                "INSERT OR IGNORE INTO budget VALUES (1, 0, 0, 0, NULL, NULL);"
                "CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, "
                "specification TEXT NOT NULL, reason TEXT NOT NULL, "
                "status TEXT NOT NULL, planned INTEGER NOT NULL, result TEXT);"
                "CREATE TABLE IF NOT EXISTS model_calls (id TEXT PRIMARY KEY, "
                "reservation REAL NOT NULL, "
                "cost REAL, provider TEXT NOT NULL, model TEXT NOT NULL);"
            )
        self.reconcile_interrupted()
        logger.info(
            "Search budgets: candidates<=%d; simulator calls<=%d (%d per candidate); "
            "model requests<=%d; estimated model cost<=$%.2f (not Copilot quota)",
            self.max_evaluations,
            self.max_simulations,
            self.planned_calls,
            self.max_model_calls,
            self.max_model_cost_usd,
        )
        logger.info(
            "Scores in Pa: RMSE=root mean squared error; "
            "CRPS=continuous ranked probability score (lower is better)"
        )
        logger.info(
            "Pressure-error check: conditioning and validation RMSE<=%.2f Pa; "
            "passing is not proof of geological truth",
            2 * self.case.observation_std_pa,
        )
        snapshot = self.snapshot()
        if not snapshot["closed"]:
            self.submit({"kind": "homogeneous"}, "Predeclared initial homogeneous hypothesis")
        else:
            self.finish(snapshot["outcome"], snapshot["reason"])

    def close(self) -> None:
        """Release exclusive campaign ownership after the worker stops."""
        self.owner.close()

    def __enter__(self) -> "Campaign":
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def submit(self, specification: object, reason: object) -> dict[str, object]:
        """Validate and reserve one candidate transactionally; equivalent repeats reuse its run."""
        hypothesis = parse_hypothesis(specification, self.case)
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= MAX_REASON_LENGTH:
            raise ValueError(
                f"reason must be a non-empty string of at most {MAX_REASON_LENGTH} characters"
            )
        identifier = hypothesis.identifier(self.case)
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM evaluations WHERE id = ?", (identifier,)
            ).fetchone()
            if existing is not None:
                logger.info(
                    "Equivalent proposal reused: %s (status=%s); no new simulator reservation",
                    identifier,
                    existing["status"],
                )
                logger.info("Proposal rationale for %s: %s", identifier, reason)
                return self.public_evaluation(existing)
            budget = connection.execute("SELECT * FROM budget WHERE id = 1").fetchone()
            if budget["closed"]:
                raise ValueError("campaign is closed")
            count = connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
            if count >= self.max_evaluations:
                raise ValueError("hypothesis-evaluation budget exhausted")
            if budget["spent"] + budget["reserved"] + self.planned_calls > self.max_simulations:
                raise ValueError("simulator-call budget exhausted")
            connection.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, 'queued', ?, NULL)",
                (
                    identifier,
                    json.dumps(hypothesis.specification(), sort_keys=True),
                    reason,
                    self.planned_calls,
                ),
            )
            connection.execute(
                "UPDATE budget SET reserved = reserved + ? WHERE id = 1", (self.planned_calls,)
            )
        logger.info(
            "Candidate %d/%d queued: %s; hypothesis=%s; reserved %d simulator calls",
            count + 1,
            self.max_evaluations,
            identifier,
            hypothesis.specification(),
            self.planned_calls,
        )
        logger.info("Proposal rationale for %s: %s", identifier, reason)
        return self.evaluation(identifier)

    def execute(self, identifier: str) -> dict[str, object]:
        """Serialize and claim queued work once, independently of the Pi conversation."""
        # Direct Python clients must obey the same execution limit as the HTTP worker.
        with self.execution_lock:
            with closing(self.connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM evaluations WHERE id = ?", (identifier,)
                ).fetchone()
                if row is None:
                    raise ValueError("unknown run identifier")
                if row["status"] != "queued":
                    return self.public_evaluation(row)
                connection.execute(
                    "UPDATE evaluations SET status = 'running' WHERE id = ?", (identifier,)
                )
            hypothesis = parse_hypothesis(json.loads(row["specification"]), self.case)
            executor = LocalExecutor(self.model, row["planned"])
            logger.info(
                "Candidate started: %s (%s); artifacts=%s",
                identifier,
                hypothesis.family.value,
                self.output / identifier,
            )
            try:
                result = condition_candidate(
                    hypothesis,
                    executor,
                    self.conditioning,
                    self.validation,
                    self.output / identifier,
                    self.members,
                    self.seed,
                )
            except Exception as error:
                result = {
                    "family": hypothesis.family.value,
                    "hypothesis": hypothesis.specification(),
                    "status": "failed",
                    "error": "controller execution failed; inspect ledger.sqlite for details",
                    "operator_error": repr(error),
                    "simulator_calls": executor.simulations,
                }
            self.complete(identifier, result, executor.simulations)
            if result["status"] == "succeeded":
                logger.info(
                    "Candidate complete: %s; conditioning RMSE=%.2f Pa; "
                    "validation RMSE=%.2f Pa; validation CRPS=%.2f Pa (lower is better); "
                    "pressure-error check=%s; simulator calls=%d/%d; numerical wall=%.2fs",
                    identifier,
                    result["conditioning_rmse_final_pa"],
                    result["validation"]["rmse_pa"],
                    result["validation"]["crps_pa"],
                    "passed" if self.adequate(result) else "failed",
                    executor.simulations,
                    row["planned"],
                    result["wall_seconds"],
                )
            else:
                logger.error(
                    "Candidate %s %s: %s; simulator calls=%d/%d; inspect %s and %s",
                    identifier,
                    result["status"],
                    result["error"],
                    executor.simulations,
                    row["planned"],
                    self.output / identifier,
                    self.database,
                )
            return self.evaluation(identifier)

    def complete(self, identifier: str, result: dict[str, object], charge: int) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM evaluations WHERE id = ?", (identifier,)
            ).fetchone()
            if row["status"] != "running":
                return
            connection.execute(
                "UPDATE evaluations SET status = ?, result = ? WHERE id = ?",
                (result["status"], json.dumps(result, allow_nan=False), identifier),
            )
            connection.execute(
                "UPDATE budget SET reserved = reserved - ?, spent = spent + ? WHERE id = 1",
                (row["planned"], charge),
            )

    def reconcile_interrupted(self) -> None:
        """Reconcile completed artifacts; never automatically replay interrupted work."""
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM evaluations WHERE status = 'running'"
            ).fetchall()
        for row in rows:
            path = self.output / row["id"] / "result.json"
            if path.exists():
                result = read_json(path)
                if result["hypothesis"] != json.loads(row["specification"]):
                    raise ValueError(
                        "saved result describes a different hypothesis from the recorded request"
                    )
                self.complete(row["id"], result, result["simulator_calls"])
            else:
                result = {
                    "family": parse_hypothesis(
                        json.loads(row["specification"]), self.case
                    ).family.value,
                    "hypothesis": json.loads(row["specification"]),
                    "status": "interrupted",
                    "error": (
                        "controller stopped before saving a complete result; work was not rerun"
                    ),
                    "simulator_calls": None,
                    "conservative_budget_charge": row["planned"],
                }
                self.complete(row["id"], result, row["planned"])

    def evaluation(self, identifier: str) -> dict[str, object]:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM evaluations WHERE id = ?", (identifier,)
            ).fetchone()
        if row is None:
            raise ValueError("unknown run identifier")
        return self.public_evaluation(row)

    def public_evaluation(self, row: sqlite3.Row) -> dict[str, object]:
        """Expose conditioning/validation scores and residuals without protected data."""
        result = json.loads(row["result"]) if row["result"] else None
        public_result = None
        if result is not None:
            public_result = {
                key: result[key]
                for key in (
                    "status",
                    "error",
                    "member_id",
                    "conditioning_rmse_history_pa",
                    "conditioning_rmse_final_pa",
                    "validation",
                    "parameters",
                    "diagnostics",
                    "simulator_calls",
                    "wall_seconds",
                )
                if key in result
            }
            public_result["adequacy_gate_passed"] = self.adequate(result)
        return {
            "run_id": row["id"],
            "hypothesis": json.loads(row["specification"]),
            "reason": row["reason"],
            "status": row["status"],
            "result": public_result,
        }

    def adequate(self, result: dict[str, object]) -> bool:
        """Check conditioning and validation RMSE against twice the observation-error std."""
        return result["status"] == "succeeded" and (
            result["conditioning_rmse_final_pa"] <= 2 * self.case.observation_std_pa
            and result["validation"]["rmse_pa"] <= 2 * self.case.observation_std_pa
        )

    def snapshot(self) -> dict[str, object]:
        """Return campaign state and spent, reserved, and remaining budgets."""
        with closing(self.connect()) as connection:
            budget = connection.execute("SELECT * FROM budget WHERE id = 1").fetchone()
            rows = connection.execute("SELECT * FROM evaluations ORDER BY rowid").fetchall()
            calls = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(COALESCE(cost, reservation)), 0) FROM model_calls"
            ).fetchone()
        return {
            "schema_version": 1,
            "closed": bool(budget["closed"]),
            "outcome": budget["outcome"],
            "reason": budget["reason"],
            "evaluations": [self.public_evaluation(row) for row in rows],
            "remaining_evaluations": self.max_evaluations - len(rows),
            "remaining_simulator_calls": self.max_simulations
            - budget["spent"]
            - budget["reserved"],
            "charged_simulator_calls": budget["spent"],
            "reserved_simulator_calls": budget["reserved"],
            "model_calls": calls[0],
            "remaining_model_calls": self.max_model_calls - calls[0],
            "model_cost_estimate_usd": calls[1],
            "remaining_model_cost_estimate_usd": max(0.0, self.max_model_cost_usd - calls[1]),
        }

    def reserve_model_call(self, value: object) -> dict[str, object]:
        """Reserve catalog-estimated cost before host dispatch, outside proposer tools."""
        if not isinstance(value, dict) or set(value) != {
            "call_id",
            "reservation_usd",
            "provider",
            "model",
        }:
            raise ValueError("model-call reservation has unsupported fields")
        identifier = value["call_id"]
        amount = value["reservation_usd"]
        if not isinstance(identifier, str) or not re.fullmatch(r"mc_[0-9a-f]{32}", identifier):
            raise ValueError("invalid model-call identifier")
        if type(amount) not in (int, float) or not isfinite(amount) or amount < 0:
            raise ValueError("model-cost reservation must be a finite non-negative USD amount")
        if any(
            not isinstance(value[key], str) or not 1 <= len(value[key]) <= MAX_MODEL_NAME_LENGTH
            for key in ("provider", "model")
        ):
            raise ValueError(
                "provider and model names must be text with "
                f"1 to {MAX_MODEL_NAME_LENGTH} characters"
            )
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM model_calls WHERE id = ?", (identifier,)
            ).fetchone():
                raise ValueError(
                    "model-call identifier already reserved; do not replay provider requests"
                )
            if connection.execute("SELECT closed FROM budget WHERE id = 1").fetchone()[0]:
                raise ValueError("campaign is closed")
            calls = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(COALESCE(cost, reservation)), 0) FROM model_calls"
            ).fetchone()
            if (
                calls[0] >= self.max_model_calls
                or calls[1] + amount > self.max_model_cost_usd + MODEL_COST_ROUNDING_TOLERANCE_USD
            ):
                raise ValueError("model-request or estimated-cost budget exhausted")
            connection.execute(
                "INSERT INTO model_calls VALUES (?, ?, NULL, ?, ?)",
                (identifier, amount, value["provider"], value["model"]),
            )
        logger.info(
            "Model request %d/%d: provider=%s; model=%s; "
            "estimated cost reserved=$%.6f; awaiting response (call=%s)",
            calls[0] + 1,
            self.max_model_calls,
            value["provider"],
            value["model"],
            amount,
            identifier,
        )
        return {"call_id": identifier, "reserved": True}

    def settle_model_call(self, value: object) -> dict[str, object]:
        """Settle estimated cost, retaining reservations when provider usage is unknown."""
        if not isinstance(value, dict) or set(value) != {"call_id", "cost_usd"}:
            raise ValueError("model-call settlement has unsupported fields")
        amount = value["cost_usd"]
        if amount is not None and (
            type(amount) not in (int, float) or not isfinite(amount) or amount < 0
        ):
            raise ValueError("cost must be null (unknown usage) or a finite non-negative amount")
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM model_calls WHERE id = ?", (value["call_id"],)
            ).fetchone()
            if row is None:
                raise ValueError("unknown model-call reservation")
            charge = row["reservation"] if amount is None else amount
            if row["cost"] is not None and row["cost"] != charge:
                raise ValueError("model-call cost is already settled")
            connection.execute(
                "UPDATE model_calls SET cost = ? WHERE id = ?", (charge, value["call_id"])
            )
        if row["cost"] is None:
            if amount is None:
                logger.warning(
                    "Model request %s: usage unknown; estimated cost reservation retained=$%.6f",
                    value["call_id"],
                    charge,
                )
            else:
                logger.info(
                    "Model request %s returned: estimated cost=$%.6f",
                    value["call_id"],
                    charge,
                )
        return self.snapshot()

    def public_evidence(self) -> dict[str, object]:
        """Expose controls and conditioning observations, not raw validation/final data or truth.

        Public means agent-visible. Validation scores arrive through public_evaluation.
        """
        values = self.conditioning["values_pa"].reshape(-1, len(self.case.gauge_ids))
        return {
            "schema_version": 1,
            "case_id": "pressure_diffusion_v1",
            "physics": asdict(self.case),
            "boundary_condition": "zero_flux",
            "gauge_ids": list(self.case.gauge_ids),
            "gauge_positions_fraction": self.case.gauge_positions.tolist(),
            "well_positions_fraction": self.case.well_positions.tolist(),
            "well_order": ["west", "east", "south", "north"],
            "rates_m3_s": self.case.rates.tolist(),
            "times_hours": (
                self.case.observation_steps[: values.shape[0]] * self.case.time_step_s / 3600
            ).tolist(),
            "observed_pressure_change_bar": (
                (values - self.case.initial_pressure_pa) / 1e5
            ).tolist(),
            "hypothesis_grammar": {
                "kinds": ["homogeneous", "band"],
                "orientations": ["vertical", "horizontal"],
                "band_width_bounds": [2 / self.case.grid_size, 0.5],
                "geometry_rule": (
                    "center_fraction and width_fraction are fractions of the domain length; "
                    "vertical bands span all y, horizontal bands all x; "
                    "band inside domain, at least two grid rows/columns, background on both sides"
                ),
                "conditioned_parameters": [
                    "background permeability (mD)",
                    "band/background permeability ratio, when a band is present",
                ],
                "fixed_inference": (
                    f"{self.members} ensemble members; "
                    f"{len(INFLATION)} equal-inflation ES-MDA updates; "
                    "fixed priors and independent Gaussian observation error"
                ),
            },
            "adequacy_rule": (
                "conditioning and validation RMSE <= 2 observation standard deviations; "
                "prototype error check, not proof of geological truth"
            ),
        }

    def finish(self, outcome: object, reason: object) -> dict[str, object]:
        """Freeze search before protected grading and validate adequate claims."""
        with self.publication_lock:
            return self.freeze(outcome, reason)

    def freeze(self, outcome: object, reason: object) -> dict[str, object]:
        if outcome not in ("adequate", "inadequate", "uncertain"):
            raise ValueError("outcome must be adequate, inadequate, or uncertain")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= MAX_REASON_LENGTH:
            raise ValueError(
                f"finish reason must be non-empty and at most {MAX_REASON_LENGTH} characters"
            )
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT * FROM evaluations ORDER BY rowid").fetchall()
            if any(row["status"] in ("queued", "running") for row in rows):
                raise ValueError(
                    "wait for accepted hypothesis fitting and scoring before finishing search"
                )
            records = []
            for row in rows:
                result = json.loads(row["result"])
                result["candidate_id"] = row["id"]
                records.append(result)
            ranked = sorted(
                (record for record in records if record["status"] == "succeeded"),
                key=lambda record: record["validation"]["crps_pa"],
            )
            if outcome == "adequate" and (not ranked or not self.adequate(ranked[0])):
                raise ValueError(
                    "best evaluated hypothesis fails the pressure-error check: "
                    "fitting and validation RMSE must both be <= "
                    "twice the measurement-noise standard deviation"
                )
            budget = connection.execute("SELECT * FROM budget WHERE id = 1").fetchone()
            model_rows = connection.execute(
                "SELECT DISTINCT provider, model FROM model_calls"
            ).fetchall()
            if not budget["closed"]:
                connection.execute(
                    "UPDATE budget SET closed = 1, outcome = ?, reason = ? WHERE id = 1",
                    (outcome, reason),
                )
            elif outcome != budget["outcome"] or reason != budget["reason"]:
                raise ValueError("search decision is already frozen")
        selection_path = self.output / "selection.json"
        if not selection_path.exists():
            counts = [record["simulator_calls"] for record in records]
            selection = {
                "schema_version": 1,
                "status": "succeeded" if ranked else "failed",
                "outcome": outcome,
                "reason": reason,
                "candidates": records,
                "selected_candidate_id": ranked[0]["candidate_id"] if ranked else None,
                "selected_family": ranked[0]["family"] if ranked else None,
                "validation_ranking": [record["candidate_id"] for record in ranked],
                "selection_rule": (
                    "lowest validation CRPS among tested hypotheses; adequacy checked separately"
                ),
                "simulator_calls": sum(counts)
                if all(value is not None for value in counts)
                else None,
                "charged_simulator_calls": self.snapshot()["charged_simulator_calls"],
                "wall_seconds": sum(record.get("wall_seconds", 0) for record in records),
                "final_evaluation_accessed": False,
                "model_calls": self.snapshot()["model_calls"],
                "proposer_models": [
                    {"provider": row["provider"], "model": row["model"]} for row in model_rows
                ],
                "proposer_mode": (
                    "no_model_calls"
                    if not model_rows
                    else "faux_scripted_smoke_test"
                    if all(row["provider"] == "faux" for row in model_rows)
                    else "live_llm"
                ),
                "model_cost_estimate_usd_at_finish": self.snapshot()["model_cost_estimate_usd"],
            }
            write_json(selection_path, selection)
        if not (self.output / "run.json").exists():
            write_json(
                self.output / "run.json",
                {
                    "schema_version": 1,
                    "evidence_sha256": checksum(self.evidence / "evidence.json"),
                    "selection_sha256": checksum(selection_path),
                    "implementation": implementation(),
                    "members": self.members,
                    "seed": self.seed,
                    "mode": "hypothesis_search",
                },
            )
        state = self.snapshot()
        if not budget["closed"]:
            logger.info(
                "Search frozen: outcome=%s; selected=%s; successful candidates=%d/%d",
                outcome,
                ranked[0]["candidate_id"] if ranked else "none",
                len(ranked),
                len(records),
            )
            logger.info("Search stopping reason: %s", reason)
            for position, record in enumerate(ranked, start=1):
                logger.info(
                    "Validation rank %d: %s (%s); CRPS=%.2f Pa (lower is better); "
                    "pressure-error check=%s",
                    position,
                    record["candidate_id"],
                    record["family"],
                    record["validation"]["crps_pa"],
                    "passed" if self.adequate(record) else "failed",
                )
            logger.info(
                "Search usage: model requests=%d/%d; estimated model cost=$%.6f/$%.2f "
                "(not Copilot quota); charged simulator calls=%d/%d",
                state["model_calls"],
                self.max_model_calls,
                state["model_cost_estimate_usd"],
                self.max_model_cost_usd,
                state["charged_simulator_calls"],
                self.max_simulations,
            )
            logger.info("Frozen ranking saved: %s", selection_path)
        return state
