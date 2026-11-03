"""Prepare truth/observations and grade frozen selections separately from search."""

import platform
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cases.pressure import Phase, PressureCase
from controller.artifacts import (
    checksum,
    read_arrays,
    read_json,
    write_arrays,
    write_bytes,
    write_json,
)
from controller.hypotheses import parse_hypothesis, validate_band_geometry
from controller.progress import logger
from numerical.evaluation import (
    parameter_summary,
    predictive_scores,
)
from numerical.geology import (
    BandGeometry,
    Family,
    band_cells,
    encode_parameters,
    parameter_names,
    permeability_field,
)
from numerical.pressure import PressureModel
from numerical.report import diagnostic_png


def prepare_case(
    output: Path,
    case: PressureCase,
    truth_family: Family = Family.VERTICAL_BAND,
    background_md: float = 100.0,
    band_ratio: float = 0.01,
    noise_seed: int = 2026,
    *,
    truth_geometry: BandGeometry,
) -> dict[str, str]:
    """Save conditioning/validation under evidence/ and hidden truth/final data under protected/.

    The CLI uses fixed truth defaults; programmatic callers may vary the generating family/geometry.
    """
    if type(noise_seed) is not int or noise_seed < 0:
        raise ValueError("noise_seed must be a non-negative integer")
    parameters = encode_parameters(truth_family, background_md, band_ratio)
    model = PressureModel(case)
    validate_band_geometry(truth_geometry, case)
    field = permeability_field(truth_family, parameters, case, truth_geometry)
    logger.info("Simulating synthetic pressure history: %d time steps", case.total_steps)
    truth = model.simulate(field, case.total_steps)
    rng = np.random.default_rng(noise_seed)
    observed = truth.responses_pa + rng.normal(
        0, case.observation_std_pa, size=truth.responses_pa.shape
    )
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    protected = output / "protected"
    references: dict[str, str] = {}
    for phase in Phase:
        mask = case.phase_mask(phase)
        count = int(mask.sum())
        directory = protected if phase == Phase.FINAL else evidence
        references[phase.value] = write_arrays(
            directory / f"{phase.value}.npz",
            values_pa=observed[mask],
            variance_pa2=np.full(count, case.observation_std_pa**2),
            observation_ids=case.observation_ids[mask],
        )
        logger.info(
            "Saved %s observations: %d readings",
            "protected final" if phase == Phase.FINAL else phase.value,
            count,
        )
    evidence_checksum = write_json(
        evidence / "evidence.json",
        {
            "schema_version": 1,
            "case": case.specification(),
            "implementation": implementation(),
            "conditioning_sha256": references[Phase.CONDITIONING],
            "validation_sha256": references[Phase.VALIDATION],
        },
    )
    truth_checksum = write_arrays(
        protected / "truth.npz",
        permeability_m2=field,
        clean_responses_pa=truth.responses_pa,
        final_pressure_pa=truth.final_pressure_pa,
    )
    write_json(
        protected / "truth.json",
        {
            "schema_version": 1,
            "evidence_sha256": evidence_checksum,
            "family": truth_family.value,
            "background_md": background_md,
            "band_ratio": None if truth_family == Family.HOMOGENEOUS else band_ratio,
            "geometry": None
            if truth_family == Family.HOMOGENEOUS
            else {
                "center_fraction": truth_geometry.center_fraction,
                "width_fraction": truth_geometry.width_fraction,
            },
            "noise_seed": noise_seed,
            "noise_rng_state_after": rng.bit_generator.state,
            "truth_sha256": truth_checksum,
            "final_sha256": references[Phase.FINAL],
            "truth_simulator_calls": 1,
            "max_balance_error_m3": truth.max_balance_error_m3,
        },
    )
    logger.info("Saved hidden truth; protected values are not sent to the proposer")
    return {"evidence": str(evidence), "protected": str(protected)}


def evaluate_selection(
    evidence: Path, protected: Path, run: Path, output: Path
) -> dict[str, object]:
    """Grade a frozen selection against protected truth/final observations without refitting."""
    case, manifest = load_case(evidence)
    run_manifest = read_json(run / "run.json")
    truth_manifest = read_json(protected / "truth.json")
    evidence_checksum = checksum(evidence / "evidence.json")
    if any(item["evidence_sha256"] != evidence_checksum for item in (run_manifest, truth_manifest)):
        raise ValueError(
            "run and hidden-answer files must refer to the same measurement files and settings"
        )
    if checksum(run / "selection.json") != run_manifest["selection_sha256"]:
        raise ValueError("selection has changed after the search ended")
    selection = read_json(run / "selection.json")
    if selection["status"] != "succeeded":
        raise ValueError("cannot evaluate a run without successful candidates")
    logger.info("Frozen selection verified; grading against protected truth and final observations")
    truth = read_arrays(protected / "truth.npz", truth_manifest["truth_sha256"])
    final = load_observations(protected, case, truth_manifest, Phase.FINAL)
    reports: list[dict[str, object]] = []
    selected = Family(selection["selected_family"])
    selected_id = selection["selected_candidate_id"]
    selected_posterior = None
    selected_field = None
    selected_geometry = None
    for candidate in selection["candidates"]:
        if candidate["status"] != "succeeded":
            continue
        family = Family(candidate["family"])
        hypothesis = parse_hypothesis(candidate["hypothesis"], case)
        posterior = load_posterior(run, case, candidate)
        fields = np.stack(
            [
                permeability_field(family, values, case, hypothesis.geometry)
                for values in posterior["parameters"].T
            ]
        )
        mean_log_field = np.log(fields).mean(axis=0)
        if candidate["candidate_id"] == selected_id:
            selected_posterior = posterior
            selected_field = np.exp(mean_log_field)
            selected_geometry = hypothesis.geometry
        final_scores = predictive_scores(
            posterior["predictions_pa"][case.phase_mask(Phase.FINAL)], final["values_pa"]
        )
        reports.append(
            {
                "family": family.value,
                "candidate_id": candidate["candidate_id"],
                "hypothesis": hypothesis.specification(),
                "final": final_scores,
                "log_permeability_rmse": float(
                    np.sqrt(np.mean((mean_log_field - np.log(truth["permeability_m2"])) ** 2))
                ),
                "parameters": parameter_summary(family, posterior["parameters"]),
            }
        )
        logger.info(
            "Graded candidate %s (%s): final RMSE=%.2f Pa; final CRPS=%.2f Pa",
            candidate["candidate_id"],
            family.value,
            final_scores["rmse_pa"],
            final_scores["crps_pa"],
        )
    if selected_posterior is None or selected_field is None:
        raise ValueError("selected hypothesis has no successful fitted results in posterior.npz")
    selected_report = next(report for report in reports if report["candidate_id"] == selected_id)
    geometry_recovered = selected.value == truth_manifest["family"]
    if geometry_recovered and selected_geometry is not None:
        geometry_recovered = np.array_equal(
            band_cells(case, selected_geometry),
            band_cells(case, BandGeometry(**truth_manifest["geometry"])),
        )
    parameter_recovery = None
    if geometry_recovered:
        targets = {"background_md": truth_manifest["background_md"]}
        if selected != Family.HOMOGENEOUS:
            targets["band_ratio"] = truth_manifest["band_ratio"]
        parameter_recovery = {
            name: {
                "truth": target,
                "relative_median_error": abs(summary["median"] - target) / target,
                "covered_by_90_interval": summary["q05"] <= target <= summary["q95"],
            }
            for name, target in targets.items()
            for summary in [selected_report["parameters"][name]]
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "evidence_sha256": evidence_checksum,
        "selection_sha256": run_manifest["selection_sha256"],
        "selected_family": selected.value,
        "truth_family": truth_manifest["family"],
        "family_recovered": selected.value == truth_manifest["family"],
        "structural_truth_recovered": bool(geometry_recovered),
        "selected_candidate_id": selected_id,
        "search_outcome": selection["outcome"],
        "proposer_mode": selection["proposer_mode"],
        "model_calls": selection["model_calls"],
        "model_cost_estimate_usd": selection["model_cost_estimate_usd_at_finish"],
        "proposer_models": selection["proposer_models"],
        "parameter_recovery": parameter_recovery,
        "candidates": reports,
        "simulator_calls": selection["simulator_calls"],
        "truth_simulator_calls": truth_manifest["truth_simulator_calls"],
        "wall_seconds": selection["wall_seconds"],
        "interpretation": (
            "selection only ranks the evaluated hypotheses; observed interval coverage "
            "does not establish 90% coverage across other experiments"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    observed = np.empty(case.observation_ids.size)
    for phase in Phase:
        phase_data = (
            final if phase == Phase.FINAL else load_observations(evidence, case, manifest, phase)
        )
        observed[case.phase_mask(phase)] = phase_data["values_pa"]
    logger.info("Writing final report and diagnostic plot: %s", output)
    image = diagnostic_png(
        case,
        truth["permeability_m2"],
        selected_field,
        truth["clean_responses_pa"],
        selected_posterior["predictions_pa"],
        observed,
        selected.value,
    )
    report["diagnostic_png_sha256"] = write_bytes(output / "diagnostics.png", image)
    write_json(output / "report.json", report)
    logger.info(
        "Final assessment: selected=%s; family recovered=%s; structural truth recovered=%s",
        selected_id,
        report["family_recovered"],
        report["structural_truth_recovered"],
    )
    return report


def load_case(evidence: Path) -> tuple[PressureCase, dict[str, object]]:
    """Validate evidence settings, controls, and observation ordering."""
    manifest = read_json(evidence / "evidence.json")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported measurement-settings file format version")
    case = PressureCase(**manifest["case"]["settings"])
    if manifest["case"] != case.specification():
        raise ValueError(
            "measurement settings differ from the supported well schedule or reading order"
        )
    return case, manifest


def load_observations(
    directory: Path, case: PressureCase, manifest: dict[str, object], phase: Phase
) -> dict[str, NDArray[np.generic]]:
    """Validate observation ordering, dtype, error model, and artifact checksum."""
    arrays = read_arrays(directory / f"{phase.value}.npz", manifest[f"{phase.value}_sha256"])
    expected_ids = case.observation_ids[case.phase_mask(phase)]
    if not np.array_equal(arrays["observation_ids"], expected_ids):
        raise ValueError(f"{phase.value} observation IDs/order do not match the case")
    for name in ("values_pa", "variance_pa2"):
        values = arrays[name]
        if values.shape != expected_ids.shape or values.dtype != np.float64:
            raise ValueError(
                f"{phase.value} {name} must be a float64 vector with one value per observation"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{phase.value} {name} contains NaN or infinity")
    if not (arrays["variance_pa2"] == case.observation_std_pa**2).all():
        raise ValueError("observation variance differs from the fixed error model")
    return arrays


def load_posterior(
    run: Path, case: PressureCase, candidate: dict[str, object]
) -> dict[str, NDArray[np.generic]]:
    """Load posterior artifacts, checking integrity and observation/member correspondence."""
    family = Family(candidate["family"])
    component = candidate["candidate_id"]
    if not re.fullmatch(r"h_[0-9a-f]{24}", component):
        raise ValueError("invalid hypothesis result-directory identifier")
    arrays = read_arrays(run / component / "posterior.npz", candidate["posterior_sha256"])
    if not np.array_equal(arrays["observation_ids"], case.observation_ids):
        raise ValueError(
            "posterior.npz fitted pressure readings are not in the experiment's reading order"
        )
    if not np.array_equal(arrays["parameter_names"], parameter_names(family)):
        raise ValueError("posterior.npz fitted parameter rows differ from the chosen pattern type")
    members = arrays["parameters"].shape[1]
    if not np.array_equal(
        arrays["member_ids"], [f"member_{member:04d}" for member in range(members)]
    ):
        raise ValueError(
            "posterior.npz parameter-version columns differ from the submitted member order"
        )
    for name in ("responses_pa", "predictions_pa"):
        if arrays[name].shape != (case.observation_ids.size, members):
            raise ValueError(
                "posterior.npz pressure array must have one row per reading "
                "and one column per member"
            )
    return arrays


def implementation() -> dict[str, object]:
    """Record package versions and source/lockfile checksums for reproducibility."""
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "cases/pressure.py",
        "numerical/geology.py",
        "numerical/pressure.py",
        "numerical/evaluation.py",
        "numerical/report.py",
        "controller/assimilation.py",
        "controller/candidate.py",
        "controller/hypotheses.py",
        "controller/campaign.py",
        "controller/server.py",
        "controller/agent_run.py",
        "controller/experiment.py",
        "controller/local.py",
    ]
    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("fipy", "iterative-ensemble-smoother", "numpy", "scipy", "matplotlib")
        },
        "source_sha256": {name: checksum(root / name) for name in source_paths},
        "lockfile_sha256": checksum(root / "uv.lock"),
        "solver": "FiPy finite-volume assembly, harmonic face mobility, SciPy sparse LU reuse",
        "time_integration": "backward Euler, fixed time step",
    }
