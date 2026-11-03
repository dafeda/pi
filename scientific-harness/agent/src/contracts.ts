// Defines the closed proposal and public-feedback contract; protected truth and final data are excluded.
import {
	arr,
	bool,
	compile,
	int,
	lit,
	nullable,
	num,
	obj,
	oneOf,
	opt,
	type Static,
	str,
	union,
} from "@earendil-works/pi-schema";

/** Allows homogeneous permeability or one low-permeability band. */
export const HypothesisSchema = union([
	obj({ kind: lit("homogeneous") }),
	obj({
		kind: lit("band"),
		orientation: oneOf(["vertical", "horizontal"]),
		center_fraction: num({ minimum: 0, maximum: 1 }),
		width_fraction: num({ exclusiveMinimum: 0, maximum: 0.5 }),
	}),
]);

/** Requires an allowed hypothesis and an explanation for testing it. */
export const EvaluateSchema = obj({ hypothesis: HypothesisSchema, reason: str({ minLength: 1, maxLength: 2000 }) });

/** Declares an adequate, inadequate, or uncertain search outcome. */
export const FinishSchema = obj({
	outcome: oneOf(["adequate", "inadequate", "uncertain"]),
	reason: str({ minLength: 1, maxLength: 2000 }),
});

// Pressure-curve rows are measurement times; the four columns follow gauge_ids.
const numbers = arr(num(), { maxItems: 128 });
const four = arr(num(), { minItems: 4, maxItems: 4 });
const curves = arr(four, { maxItems: 128 });
// CRPS scores the predictive ensemble; coverage/width describe its 90% interval. Pressure units are Pa.
const scores = obj({ crps_pa: num(), rmse_pa: num(), coverage_90: num(), interval_width_90_pa: num() });

// q05/q95 are posterior 5th/95th percentiles across ensemble members.
const statistics = obj({ mean: num(), median: num(), q05: num(), q95: num() });
const diagnostics = obj({
	residual_definition: str(),
	gauge_ids: arr(str(), { minItems: 4, maxItems: 4 }),
	times_hours: numbers,
	observed_pressure_change_bar: curves,
	predicted_pressure_change_bar: curves,
	signed_residual_std: curves,
	periods: arr(obj({ period: str(), mean_signed_residual_std: four, rmse_std: four }), { maxItems: 4 }),
});
const result = obj({
	status: oneOf(["succeeded", "failed", "interrupted"]),
	error: opt(str()),
	member_id: opt(nullable(str())),
	conditioning_rmse_history_pa: opt(numbers),
	conditioning_rmse_final_pa: opt(num()),
	validation: opt(scores),
	parameters: opt(obj({ background_md: statistics, band_ratio: opt(statistics), band_md: opt(statistics) })),
	diagnostics: opt(diagnostics),
	simulator_calls: opt(nullable(int({ minimum: 0 }))),
	wall_seconds: opt(num()),
	adequacy_gate_passed: bool(),
});

/** Describes one queued or completed hypothesis evaluation. */
export const EvaluationSchema = obj({
	run_id: str({ pattern: "^h_[0-9a-f]{24}$" }),
	hypothesis: HypothesisSchema,
	reason: str(),
	status: oneOf(["queued", "running", "succeeded", "failed", "interrupted"]),
	result: nullable(result),
});
/** Describes campaign state, including evaluations and spent/reserved/remaining budgets. */
export const SnapshotSchema = obj({
	schema_version: lit(1),
	closed: bool(),
	outcome: nullable(oneOf(["adequate", "inadequate", "uncertain"])),
	reason: nullable(str()),
	evaluations: arr(EvaluationSchema, { maxItems: 32 }),
	remaining_evaluations: int(),
	remaining_simulator_calls: int(),
	charged_simulator_calls: int(),
	reserved_simulator_calls: int(),
	model_calls: int(),
	remaining_model_calls: int(),
	model_cost_estimate_usd: num(),
	remaining_model_cost_estimate_usd: num(),
});
/** Exposes controls and conditioning observations, not raw validation/final data or truth. */
export const EvidenceSchema = obj({
	schema_version: lit(1),
	case_id: lit("pressure_diffusion_v1"),
	physics: obj({
		grid_size: int(),
		length_m: num(),
		thickness_m: num(),
		storage_per_pa: num(),
		viscosity_pa_s: num(),
		initial_pressure_pa: num(),
		time_step_s: num(),
		segment_steps: int(),
		sample_every: int(),
		rate_m3_s: num(),
		observation_std_pa: num(),
	}),
	boundary_condition: lit("zero_flux"),
	gauge_ids: arr(str(), { minItems: 4, maxItems: 4 }),
	gauge_positions_fraction: arr(arr(num(), { minItems: 2, maxItems: 2 }), { minItems: 4, maxItems: 4 }),
	well_positions_fraction: arr(arr(num(), { minItems: 2, maxItems: 2 }), { minItems: 4, maxItems: 4 }),
	well_order: arr(str(), { minItems: 4, maxItems: 4 }),
	rates_m3_s: arr(four, { maxItems: 512 }),
	times_hours: numbers,
	observed_pressure_change_bar: curves,
	hypothesis_grammar: obj({
		kinds: arr(str()),
		orientations: arr(str()),
		band_width_bounds: arr(num(), { minItems: 2, maxItems: 2 }),
		geometry_rule: str(),
		conditioned_parameters: arr(str()),
		fixed_inference: str(),
	}),
	adequacy_rule: str(),
});

export type Hypothesis = Static<typeof HypothesisSchema>;
export type Evaluation = Static<typeof EvaluationSchema>;
export type Snapshot = Static<typeof SnapshotSchema>;
export type Evidence = Static<typeof EvidenceSchema>;

export const checkEvaluation = compile(EvaluationSchema);
export const checkSnapshot = compile(SnapshotSchema);
export const checkEvidence = compile(EvidenceSchema);
