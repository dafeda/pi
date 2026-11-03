"""Own queued evaluations, transactional budget reservations, and frozen search decisions."""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from cases.pressure import Phase
from controller.artifacts import checksum, write_json
from controller.assimilation import INFLATION
from controller.candidate import condition_candidate
from controller.experiment import load_case, load_observations
from controller.hypotheses import Hypothesis, parse_hypothesis
from controller.local import LocalExecutor
from controller.progress import ProgressSection, logger
from numerical.geology import Family
from numerical.pressure import PressureModel

# Bound local numerical work independently of operator-selected search depth.
MAX_GRID_SIZE = 64
MAX_TIME_STEPS = 512
MAX_CONDITIONING_TIME_SAMPLES = 128

MAX_MODEL_NAME_LENGTH = 200


class ModelBudgetExceeded(ValueError):
    """Distinguish a normal model-request limit from an unexpected host tracking failure."""


class Campaign:
    """Run one hypothesis search under fixed conditioning policy and budgets.

    Conditioning observations fit parameters; validation observations rank candidates.
    Each campaign requires a new output directory; ledger.sqlite records evaluations and requests.
    """

    def __init__(
        self,
        evidence: Path,
        output: Path,
        realizations: int = 50,
        seed: int = 7,
        max_evaluations: int = 3,
        max_model_calls: int = 6,
    ) -> None:
        for name, value, minimum in (
            ("realizations", realizations, 2),
            ("seed", seed, 0),
            ("max_evaluations", max_evaluations, 1),
            ("max_model_calls", max_model_calls, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
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
        self.realizations = realizations
        self.seed = seed
        self.max_model_calls = max_model_calls
        self.planned_calls = (len(INFLATION) + 1) * realizations
        self.max_simulations = self.planned_calls * max_evaluations
        self.max_evaluations = max_evaluations
        self.model = PressureModel(self.case)
        settings = {
            "schema_version": 1,
            "evidence_sha256": checksum(evidence / "evidence.json"),
            "realizations": realizations,
            "seed": seed,
            "max_simulations": self.max_simulations,
            "max_evaluations": max_evaluations,
            "max_model_calls": max_model_calls,
            "adequacy_rule": (
                "conditioning and validation RMSE <= 2 observation standard deviations"
            ),
        }
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "campaign.json", settings)
        self.database = output / "ledger.sqlite"

        # Persist simulator-call spending, reservations, and the frozen search decision.
        create_budget_sql = (
            "CREATE TABLE budget (id INTEGER PRIMARY KEY, "
            "spent INTEGER NOT NULL, "
            "reserved INTEGER NOT NULL, closed INTEGER NOT NULL, outcome TEXT, reason TEXT);"
        )

        # Start this run with no spending, reservations, or frozen decision.
        initialize_budget_sql = "INSERT INTO budget VALUES (1, 0, 0, 0, NULL, NULL);"

        # Persist hypothesis specifications, execution status, planned simulator calls, and results.
        create_evaluations_sql = (
            "CREATE TABLE evaluations (id TEXT PRIMARY KEY, "
            "specification TEXT NOT NULL, reason TEXT NOT NULL, "
            "status TEXT NOT NULL, planned INTEGER NOT NULL, result TEXT);"
        )

        # Track model request identities and available token usage.
        create_model_calls_sql = (
            "CREATE TABLE model_calls (id TEXT PRIMARY KEY, "
            "provider TEXT NOT NULL, model TEXT NOT NULL, usage TEXT);"
        )

        with closing(self.connect()) as connection, connection:
            connection.executescript(
                create_budget_sql
                + initialize_budget_sql
                + create_evaluations_sql
                + create_model_calls_sql
            )
        logger.info(
            "Search budgets: candidates<=%d; simulator calls<=%d (%d per candidate); "
            "model requests<=%d",
            self.max_evaluations,
            self.max_simulations,
            self.planned_calls,
            self.max_model_calls,
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
        self.submit(Hypothesis(Family.HOMOGENEOUS), "Predeclared initial homogeneous hypothesis")

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def submit(self, hypothesis: Hypothesis, reason: str) -> dict[str, object]:
        """Reserve one parsed candidate transactionally; equivalent repeats reuse its run."""
        if not reason.strip():
            raise ValueError("reason must be a non-empty string")
        identifier = hypothesis.identifier(self.case)
        details = (f"ID: {identifier}",)
        if hypothesis.geometry is None:
            details += ("Structure: homogeneous permeability",)
        else:
            geometry = hypothesis.geometry
            orientation = hypothesis.specification()["orientation"]
            axis = "x" if orientation == "vertical" else "y"
            details += (
                f"Structure: {orientation} low-permeability band",
                f"Center: {geometry.center_fraction}; width: {geometry.width_fraction} "
                "(domain fractions)",
                f"Extent: {axis}={geometry.center_fraction - geometry.width_fraction / 2:.6g}"
                f"..{geometry.center_fraction + geometry.width_fraction / 2:.6g}",
            )
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM evaluations WHERE id = ?", (identifier,)
            ).fetchone()
            if existing is not None:
                logger.info(
                    ProgressSection(
                        "PROPOSAL REUSED",
                        (
                            *details,
                            f"Existing status: {existing['status']}; "
                            "no new simulator reservation or numerical evaluation",
                        ),
                        rationale=reason,
                    )
                )
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
        title = "NEW HYPOTHESIS"
        if count == 0:
            title = "PREDECLARED BASELINE"
            details += ("Source: predeclared; not an LLM proposal",)
        logger.info(
            ProgressSection(
                f"{title} - CANDIDATE {count + 1}/{self.max_evaluations}",
                (
                    *details,
                    f"Numerical evaluation queued; {self.planned_calls} simulator calls reserved",
                ),
                rationale=reason,
            )
        )
        return self.evaluation(identifier)

    def execute(self, identifier: str) -> dict[str, object]:
        """Fit and score one reserved hypothesis inline, reusing completed evaluations."""
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
        result = condition_candidate(
            hypothesis,
            executor,
            self.conditioning,
            self.validation,
            self.output / identifier,
            self.realizations,
            self.seed,
        )
        self.complete(identifier, result, executor.simulations)
        pressure_check = "PASSED" if self.adequate(result) else "FAILED"
        logger.info(
            ProgressSection(
                "CANDIDATE RESULT",
                (
                    f"ID: {identifier}",
                    f"Conditioning RMSE: {result['conditioning_rmse_final_pa']:.2f} Pa",
                    f"Validation RMSE: {result['validation']['rmse_pa']:.2f} Pa",
                    f"Validation CRPS: {result['validation']['crps_pa']:.2f} Pa (lower is better)",
                    f"Pressure-error check: {pressure_check}",
                    f"Simulator calls: {executor.simulations}/{row['planned']}; "
                    f"numerical wall: {result['wall_seconds']:.2f}s",
                ),
            )
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
                    "realization_id",
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
        """Require conditioning and validation RMSE <= twice the observation-error std.

        The 2x cutoff is a heuristic allowance for noise and model mismatch,
        not a statistically calibrated acceptance test.
        """
        return result["status"] == "succeeded" and (
            result["conditioning_rmse_final_pa"] <= 2 * self.case.observation_std_pa
            and result["validation"]["rmse_pa"] <= 2 * self.case.observation_std_pa
        )

    def snapshot(self) -> dict[str, object]:
        """Return campaign state and spent, reserved, and remaining budgets."""
        with closing(self.connect()) as connection:
            budget = connection.execute("SELECT * FROM budget WHERE id = 1").fetchone()
            rows = connection.execute("SELECT * FROM evaluations ORDER BY rowid").fetchall()
            calls = connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0]
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
            "model_calls": calls,
            "remaining_model_calls": self.max_model_calls - calls,
        }

    def record_model_call(self, value: object) -> None:
        """Record and count a model request transactionally before host dispatch."""
        if not isinstance(value, dict) or set(value) != {
            "call_id",
            "provider",
            "model",
        }:
            raise ValueError("model-call record has unsupported fields")
        identifier = value["call_id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"mc_[0-9a-f]{32}", identifier):
            raise ValueError("invalid model-call identifier")
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
                    "model-call identifier already recorded; do not replay provider requests"
                )
            if connection.execute("SELECT closed FROM budget WHERE id = 1").fetchone()[0]:
                raise ValueError("campaign is closed")
            calls = connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0]
            if calls >= self.max_model_calls:
                raise ModelBudgetExceeded("model-request budget exhausted")
            connection.execute(
                "INSERT INTO model_calls VALUES (?, ?, ?, NULL)",
                (identifier, value["provider"], value["model"]),
            )
        logger.info(
            ProgressSection(
                f"LLM REQUEST {calls + 1}/{self.max_model_calls}",
                (
                    f"Model: {value['provider']} / {value['model']}",
                    "Awaiting decision: propose a hypothesis or finish search",
                    f"Call: {identifier}",
                ),
            )
        )

    def record_model_usage(self, identifier: str, usage: dict[str, object] | None) -> None:
        """Record available token usage without changing the model-request count."""
        usage_json = (
            json.dumps(usage, sort_keys=True, allow_nan=False) if usage is not None else None
        )
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT usage FROM model_calls WHERE id = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown model-call identifier")
            if row["usage"] is not None and row["usage"] != usage_json:
                raise ValueError("model-call usage is already recorded")
            connection.execute(
                "UPDATE model_calls SET usage = ? WHERE id = ?",
                (usage_json, identifier),
            )

    def public_evidence(self) -> dict[str, object]:
        """Expose controls and conditioning observations, not raw validation/final data or truth.

        Public means agent-visible. Validation scores arrive through public_evaluation.
        """
        values = self.conditioning["values_pa"].reshape(-1, len(self.case.gauge_ids))
        return {
            "schema_version": 1,
            "case_id": "pressure_diffusion_v1",
            "physics": self.case.settings(),
            "boundary_condition": "zero_flux",
            "gauge_ids": list(self.case.gauge_ids),
            "gauge_positions_fraction": self.case.gauge_positions.tolist(),
            "well_positions_fraction": self.case.well_positions.tolist(),
            "well_order": ["west", "east", "south", "north"],
            "rates_m3_s": self.case.rates.tolist(),
            "times_hours": (self.case.observation_times_s[: values.shape[0]] / 3600).tolist(),
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
                    f"{self.realizations} ensemble realizations; "
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
        if outcome not in ("adequate", "inadequate", "uncertain"):
            raise ValueError("outcome must be adequate, inadequate, or uncertain")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("finish reason must be a non-empty string")
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            budget = connection.execute("SELECT * FROM budget WHERE id = 1").fetchone()
            if budget["closed"]:
                if outcome != budget["outcome"] or reason != budget["reason"]:
                    raise ValueError("search decision is already frozen")
                return self.snapshot()
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
            model_rows = connection.execute(
                "SELECT DISTINCT provider, model FROM model_calls"
            ).fetchall()
            connection.execute(
                "UPDATE budget SET closed = 1, outcome = ?, reason = ? WHERE id = 1",
                (outcome, reason),
            )
        state = self.snapshot()
        selection_path = self.output / "selection.json"
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
            "simulator_calls": sum(record["simulator_calls"] for record in records),
            "charged_simulator_calls": state["charged_simulator_calls"],
            "wall_seconds": sum(record.get("wall_seconds", 0) for record in records),
            "final_evaluation_accessed": False,
            "model_calls": state["model_calls"],
            "proposer_models": [
                {"provider": row["provider"], "model": row["model"]} for row in model_rows
            ],
            "proposer_mode": "live_llm" if model_rows else "no_model_calls",
        }
        write_json(selection_path, selection)
        write_json(
            self.output / "run.json",
            {
                "schema_version": 1,
                "evidence_sha256": checksum(self.evidence / "evidence.json"),
                "selection_sha256": checksum(selection_path),
                "realizations": self.realizations,
                "seed": self.seed,
                "mode": "hypothesis_search",
            },
        )
        logger.info(
            ProgressSection(
                "SEARCH DECISION",
                (
                    f"Search frozen; outcome: {outcome}",
                    f"Selected: {ranked[0]['candidate_id'] if ranked else 'none'}",
                    f"Successful candidates: {len(ranked)}/{len(records)}",
                ),
                rationale=reason,
            )
        )
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
            "Search usage: model requests=%d/%d; charged simulator calls=%d/%d",
            state["model_calls"],
            self.max_model_calls,
            state["charged_simulator_calls"],
            self.max_simulations,
        )
        logger.info("Frozen ranking saved: %s", selection_path)
        return state
