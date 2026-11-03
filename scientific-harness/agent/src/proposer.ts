// Runs bounded Pi hypothesis search using public evidence and completed numerical evaluations.

import { randomUUID } from "node:crypto";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { Agent, type AgentMessage, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, AssistantMessage, Model, Models } from "@earendil-works/pi-ai";
import { EvaluateSchema, type Evaluation, FinishSchema, type Snapshot } from "./contracts.ts";
import type { Controller } from "./controller.ts";

const SYSTEM_PROMPT = `You propose geological hypotheses for a pressure-diffusion inverse problem.
Permeability measures how easily fluid flows; the initial hypothesis is homogeneous, with the same permeability in every cell.
It may be adequate, inadequate, or uncertain.
"Public evidence" means information released to you, not information published online.
You receive conditioning observations, gauge locations/times, and known well controls. Positive rates inject fluid; negative rates produce it.
Conditioning observations fit parameters; validation observations only score forecasts.
You receive validation scores, not their raw readings. Protected final observations and generating truth stay hidden.
A signed residual is (mean simulated pressure - measured pressure) / measurement-noise standard deviation.
Positive residuals indicate overprediction; negative residuals indicate underprediction. Pressure changes use bar; permeability summaries use mD.
Use observations, known controls, and signed residuals to explain mismatches before proposing a revision.
The reviewed hypothesis grammar permits homogeneous permeability or one low-permeability band.
Propose orientation, center_fraction, and width_fraction. Vertical bands span all y; horizontal bands span all x.
Center and width are domain fractions along x for vertical bands or y for horizontal bands. Follow width bounds and geometry_rule.
Geometry stays fixed during conditioning. The controller samples an ensemble from fixed priors and runs four
ES-MDA updates (Ensemble Smoother with Multiple Data Assimilation), fitting background permeability and contrast (band/background ratio).
Do not change solver, priors, likelihood, observation error, conditioning cutoffs, or scoring.
Validation ranks predictive ensembles by CRPS (continuous ranked probability score); lower is better.
The adequacy gate requires both conditioning and validation RMSE <= 2 declared observation standard deviations.
Adequacy is not proof of geological truth; retain uncertainty when evidence is insufficient.
You have no filesystem, shell, Python, protected observations, truth labels, or truth-generation code.
Request exactly one action per turn: evaluate_hypothesis or finish_search. Explain the evidence for it.
Avoid equivalent repeats with the same orientation and covered cells. Use remaining budgets and adequacy diagnostics to decide when to stop.
If no supported hypothesis is adequate, finish as inadequate or uncertain rather than declaring the least-bad model true.`;

export interface ProposerOptions {
	controller: Controller;
	models: Models;
	model: Model<Api>;
	auditDirectory: string;
	maxTurns: number;
	maxOutputTokens: number;
}

/** Runs Pi's tool loop; the model proposes hypotheses and the controller owns conditioning. */
export async function runProposer(options: ProposerOptions): Promise<Snapshot> {
	const { controller, models, model } = options;
	if (
		!Number.isInteger(options.maxTurns) ||
		options.maxTurns < 1 ||
		!Number.isInteger(options.maxOutputTokens) ||
		options.maxOutputTokens < 1 ||
		options.maxOutputTokens > model.maxTokens
	) {
		throw new Error("Invalid turn/output-token limits");
	}
	await mkdir(options.auditDirectory, { recursive: false });
	const evidence = await controller.evidence();
	let actionTaken = false;
	let currentCall: string | undefined;
	const evaluate: AgentTool<typeof EvaluateSchema, Evaluation> = {
		name: "evaluate_hypothesis",
		label: "Evaluate hypothesis",
		description:
			"Validate and queue one reviewed geological hypothesis for controller-owned conditioning and validation. The host waits for completion before the next turn.",
		parameters: EvaluateSchema,
		executionMode: "sequential",
		execute: async (_id, params, signal) => {
			const result = await controller.evaluate(params.hypothesis, params.reason, signal);
			return { content: [{ type: "text", text: JSON.stringify(result) }], details: result, terminate: true };
		},
	};
	const finish: AgentTool<typeof FinishSchema, Snapshot> = {
		name: "finish_search",
		label: "Finish search",
		description:
			"Freeze selection before protected evaluation. The controller checks adequate claims against conditioning and validation error thresholds.",
		parameters: FinishSchema,
		executionMode: "sequential",
		execute: async (_id, params, signal) => {
			const result = await controller.finish(params.outcome, params.reason, signal);
			return { content: [{ type: "text", text: JSON.stringify(result) }], details: result, terminate: true };
		},
	};
	const agent = new Agent({
		initialState: { systemPrompt: SYSTEM_PROMPT, model, thinkingLevel: "medium", tools: [evaluate, finish] },
		streamFn: (requestModel, context, streamOptions) =>
			models.streamSimple(requestModel, context, {
				...streamOptions,
				maxTokens: options.maxOutputTokens,
				maxRetries: 0,
				timeoutMs: 60000,
				cacheRetention: "none",
			}),
		toolExecution: "sequential",
		beforeToolCall: async () => {
			if (actionTaken)
				return {
					block: true,
					reason: "Only one tool action (evaluate_hypothesis or finish_search) is accepted per turn",
					terminate: true,
				};
			actionTaken = true;
			return undefined;
		},
		finishTurn: () => ({ action: "end" }),
	});
	agent.subscribe(async (event) => {
		if (event.type !== "message_end") return;
		let message: AgentMessage | object = event.message;
		if (event.message.role === "assistant") {
			const failed = event.message.stopReason === "error" || event.message.stopReason === "aborted";
			if (currentCall) {
				await controller.request("/v1/model_calls/settle", {
					call_id: currentCall,
					cost_usd: failed && event.message.usage.totalTokens === 0 ? null : event.message.usage.cost.total,
				});
				currentCall = undefined;
			}
			if (failed)
				message = {
					role: "assistant",
					provider: model.provider,
					model: model.id,
					stopReason: event.message.stopReason,
					error: "provider request failed; raw provider error omitted",
					usage: event.message.usage,
				};
		}
		await appendFile(join(options.auditDirectory, "messages.jsonl"), `${JSON.stringify(message)}\n`);
	});
	let snapshot = await controller.waitForAcceptedWork();
	let turns = 0;
	while (!snapshot.closed && turns < options.maxTurns && snapshot.remaining_model_calls > 0) {
		const prompt = `Public evidence and current campaign state:\n${JSON.stringify({ evidence, campaign: snapshot })}\nChoose one tool action and explain why. Truth and protected final evaluation remain unavailable.`;
		const inputBytes = Buffer.byteLength(
			JSON.stringify({ messages: agent.state.messages, prompt, tools: [evaluate, finish] }),
			"utf8",
		);
		if (inputBytes > 100000) break;
		// Conservative input/protocol estimate; catalog prices are not provider bills.
		const inputTokenBound = Math.min(model.contextWindow, inputBytes * 2 + 8192);
		currentCall = `mc_${randomUUID().replaceAll("-", "")}`;
		try {
			await controller.request("/v1/model_calls/reserve", {
				call_id: currentCall,
				reservation_usd: costReservation(model, options.maxOutputTokens, inputTokenBound),
				provider: model.provider,
				model: model.id,
			});
		} catch {
			currentCall = undefined;
			break;
		}
		actionTaken = false;
		turns += 1;
		await agent.prompt(prompt);
		snapshot = await controller.waitForAcceptedWork();
		const lastAssistant = agent.state.messages.findLast(
			(message): message is AssistantMessage => message.role === "assistant",
		);
		if (lastAssistant?.stopReason === "error" || lastAssistant?.stopReason === "aborted") break;
	}
	if (!snapshot.closed)
		snapshot = await controller.finish(
			"uncertain",
			"Host stopped at its request, context, estimated-cost, or provider-error limit",
		);
	await writeFile(
		join(options.auditDirectory, "summary.json"),
		JSON.stringify(
			{
				provider: model.provider,
				model: model.id,
				turns,
				maxOutputTokens: options.maxOutputTokens,
				campaign: snapshot,
			},
			null,
			2,
		),
		{ flag: "wx" },
	);
	return snapshot;
}

/** Bounds a request's cost using the highest catalog tier; estimates are not provider bills. */
export function costReservation(model: Model<Api>, outputTokens: number, inputTokens = model.contextWindow): number {
	const rates = [model.cost, ...(model.cost.tiers ?? [])];
	const input = Math.max(...rates.flatMap((rate) => [rate.input * 2, rate.cacheRead, rate.cacheWrite]));
	const output = Math.max(...rates.map((rate) => rate.output));
	return (inputTokens * input + outputTokens * output) / 1_000_000;
}
