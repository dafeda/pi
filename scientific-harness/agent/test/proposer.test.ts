// Checks one tool call per turn, rejected hidden fields, clear instructions, and safe error records without live model calls.
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createModels, fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai";
import { compile } from "@earendil-works/pi-schema";
import { EvaluateSchema, type Evaluation, type Evidence, FinishSchema, type Snapshot } from "../src/contracts.ts";
import { Controller } from "../src/controller.ts";
import { costReservation, runProposer } from "../src/proposer.ts";

const firstId = `h_${"a".repeat(24)}`;
const secondId = `h_${"b".repeat(24)}`;

const EVIDENCE: Evidence = {
	schema_version: 1,
	case_id: "pressure_diffusion_v1",
	physics: {
		grid_size: 20,
		length_m: 100,
		thickness_m: 10,
		storage_per_pa: 1e-9,
		viscosity_pa_s: 1e-3,
		initial_pressure_pa: 2e7,
		time_step_s: 3600,
		segment_steps: 8,
		sample_every: 2,
		rate_m3_s: 1e-4,
		observation_std_pa: 1000,
	},
	boundary_condition: "zero_flux",
	gauge_ids: ["sw", "se", "nw", "ne"],
	gauge_positions_fraction: [
		[0.3, 0.3],
		[0.7, 0.3],
		[0.3, 0.7],
		[0.7, 0.7],
	],
	well_positions_fraction: [
		[0.2, 0.25],
		[0.8, 0.25],
		[0.25, 0.2],
		[0.25, 0.8],
	],
	well_order: ["west", "east", "south", "north"],
	rates_m3_s: [],
	times_hours: [],
	observed_pressure_change_bar: [],
	hypothesis_grammar: {
		kinds: ["homogeneous", "band"],
		orientations: ["vertical", "horizontal"],
		band_width_bounds: [0.1, 0.5],
		geometry_rule: "strip inside the grid, with background on both sides",
		conditioned_parameters: ["background permeability", "band/background permeability ratio"],
		fixed_inference: "fixed parameter-fitting rules",
	},
	adequacy_rule: "fitting and validation pressure errors must be at most twice the measurement-noise scale",
};

function completed(id: string, adequate: boolean): Evaluation {
	return {
		run_id: id,
		hypothesis: { kind: "homogeneous" },
		reason: "fixture",
		status: "succeeded",
		result: {
			status: "succeeded",
			conditioning_rmse_final_pa: adequate ? 1000 : 30000,
			validation: {
				crps_pa: adequate ? 500 : 20000,
				rmse_pa: adequate ? 1000 : 30000,
				coverage_90: 0.9,
				interval_width_90_pa: 3000,
			},
			adequacy_gate_passed: adequate,
			simulator_calls: 250,
		},
	};
}

class FixtureController extends Controller {
	state: Snapshot = {
		schema_version: 1,
		closed: false,
		outcome: null,
		reason: null,
		evaluations: [completed(firstId, false)],
		remaining_evaluations: 2,
		remaining_simulator_calls: 500,
		charged_simulator_calls: 250,
		reserved_simulator_calls: 0,
		model_calls: 0,
		remaining_model_calls: 6,
		model_cost_estimate_usd: 0,
		remaining_model_cost_estimate_usd: 5,
	};
	evidenceValue: object = EVIDENCE;
	accepted = 0;
	waits = 0;

	constructor() {
		super("http://127.0.0.1:8910/");
	}

	async request(path: string, body?: unknown): Promise<unknown> {
		if (path === "/v1/evidence") return structuredClone(this.evidenceValue);
		if (path === "/v1/campaign") return structuredClone(this.state);
		if (path === "/v1/model_calls/reserve") {
			this.state.model_calls += 1;
			this.state.remaining_model_calls -= 1;
			return { reserved: true };
		}
		if (path === "/v1/model_calls/settle") return structuredClone(this.state);
		if (path === "/v1/evaluations") {
			const validator = compile(EvaluateSchema);
			assert.ok(validator.check(body));
			this.accepted += 1;
			const item: Evaluation = {
				run_id: secondId,
				hypothesis: body.hypothesis,
				reason: body.reason,
				status: "queued",
				result: null,
			};
			this.state.evaluations.push(item);
			this.state.remaining_evaluations -= 1;
			return structuredClone(item);
		}
		if (path.endsWith("/wait")) {
			this.waits += 1;
			const item = this.state.evaluations[1];
			item.status = "succeeded";
			item.result = completed(secondId, true).result;
			return structuredClone(item);
		}
		if (path === "/v1/finish") {
			const validator = compile(FinishSchema);
			assert.ok(validator.check(body));
			this.state.closed = true;
			this.state.outcome = body.outcome;
			this.state.reason = body.reason;
			return structuredClone(this.state);
		}
		throw new Error("Unexpected fixture operation");
	}
}

test("one accepted action per Pi turn, then host wait before the next LLM request", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-proposer-"));
	try {
		const controller = new FixtureController();
		const models = createModels();
		const faux = fauxProvider();
		models.setProvider(faux.provider);
		const proposal = {
			hypothesis: { kind: "band", orientation: "vertical", center_fraction: 0.5, width_fraction: 0.1 },
			reason: "fixture proposal",
		};
		faux.setResponses([
			fauxAssistantMessage(
				[fauxToolCall("evaluate_hypothesis", proposal), fauxToolCall("evaluate_hypothesis", proposal)],
				{ stopReason: "toolUse" },
			),
			(context) => {
				const input = context.messages.findLast((message) => message.role === "user");
				assert.ok(JSON.stringify(input).includes("adequacy_gate_passed"));
				assert.equal(controller.waits, 1);
				return fauxAssistantMessage(
					fauxToolCall("finish_search", { outcome: "adequate", reason: "feedback received" }),
					{ stopReason: "toolUse" },
				);
			},
		]);
		const state = await runProposer({
			controller,
			models,
			model: faux.getModel(),
			auditDirectory: join(directory, "audit"),
			maxTurns: 6,
			maxOutputTokens: 1024,
		});
		assert.equal(state.closed, true);
		assert.equal(controller.accepted, 1);
		assert.equal(faux.state.callCount, 2);
		assert.equal(state.model_calls, 2);
		const transcript = await readFile(join(directory, "audit/messages.jsonl"), "utf8");
		assert.ok(transcript.includes("Only one tool action"));
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("model instructions define measurements, fitting, and scores without revealing hidden answers", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-proposer-"));
	try {
		const controller = new FixtureController();
		const models = createModels();
		const faux = fauxProvider();
		models.setProvider(faux.provider);
		faux.setResponses([
			(context) => {
				const instructions = JSON.stringify(context.messages.find((message) => message.role === "system"));
				assert.ok(instructions.includes("information released to you, not information published online"));
				assert.ok(instructions.includes("validation scores, not their raw readings"));
				assert.ok(instructions.includes("mean simulated pressure - measured pressure"));
				assert.ok(instructions.includes("Ensemble Smoother with Multiple Data Assimilation"));
				assert.ok(instructions.includes("CRPS (continuous ranked probability score)"));
				assert.ok(instructions.includes("both conditioning and validation RMSE"));
				assert.ok(instructions.includes("Adequacy is not proof of geological truth"));
				assert.ok(!instructions.includes("truth_family"));
				return fauxAssistantMessage(
					fauxToolCall("finish_search", { outcome: "uncertain", reason: "Definitions checked" }),
					{ stopReason: "toolUse" },
				);
			},
		]);
		const state = await runProposer({
			controller,
			models,
			model: faux.getModel(),
			auditDirectory: join(directory, "audit"),
			maxTurns: 1,
			maxOutputTokens: 1024,
		});
		assert.equal(state.reason, "Definitions checked");
		assert.equal(controller.accepted, 0);
		assert.equal(faux.state.callCount, 1);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("unexpected protected fields are rejected before any model call", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-proposer-"));
	try {
		const controller = new FixtureController();
		controller.evidenceValue = { ...EVIDENCE, truth_family: "forbidden" };
		const models = createModels();
		const faux = fauxProvider();
		models.setProvider(faux.provider);
		await assert.rejects(
			runProposer({
				controller,
				models,
				model: faux.getModel(),
				auditDirectory: join(directory, "audit"),
				maxTurns: 2,
				maxOutputTokens: 1024,
			}),
			/public-evidence response schema/,
		);
		assert.equal(faux.state.callCount, 0);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("provider errors stop the loop without writing raw credential-bearing error text", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-proposer-"));
	try {
		const controller = new FixtureController();
		const models = createModels();
		const faux = fauxProvider();
		models.setProvider(faux.provider);
		faux.setResponses([fauxAssistantMessage([], { stopReason: "error", errorMessage: "SECRET_PROVIDER_ERROR" })]);
		const result = await runProposer({
			controller,
			models,
			model: faux.getModel(),
			auditDirectory: join(directory, "audit"),
			maxTurns: 6,
			maxOutputTokens: 1024,
		});
		assert.equal(result.outcome, "uncertain");
		assert.equal(faux.state.callCount, 1);
		assert.ok(!(await readFile(join(directory, "audit/messages.jsonl"), "utf8")).includes("SECRET_PROVIDER_ERROR"));
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("cost reservation includes the most expensive catalog tier", () => {
	const model = fauxProvider().getModel();
	const cost = {
		input: 1,
		output: 2,
		cacheRead: 0,
		cacheWrite: 1,
		tiers: [{ inputTokensAbove: 100, input: 3, output: 6, cacheRead: 1, cacheWrite: 4 }],
	};
	assert.equal(costReservation({ ...model, contextWindow: 1000, cost }, 100), 0.0066);
});
