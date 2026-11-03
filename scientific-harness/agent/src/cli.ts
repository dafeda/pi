// Configures a Copilot model or scripted faux test mode; never silently substitutes test responses.

import { parseArgs } from "node:util";
import { createModels, fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai";
import { githubCopilotProvider } from "@earendil-works/pi-ai/providers/github-copilot";
import { Controller } from "./controller.ts";
import { CopilotLoginError, resolveCopilotLogin } from "./copilot-login.ts";
import { runProposer } from "./proposer.ts";

class ConfigurationError extends Error {}

async function main(): Promise<void> {
	const { values } = parseArgs({
		options: {
			url: { type: "string", default: "http://127.0.0.1:8910/" },
			"audit-dir": { type: "string", default: "../runs/proposer" },
			model: { type: "string" },
			faux: { type: "boolean", default: false },
			"faux-orientation": { type: "string", default: "vertical" },
			"max-turns": { type: "string", default: "6" },
			"max-output-tokens": { type: "string", default: "1024" },
		},
	});
	const controller = new Controller(values.url);
	if (values.faux) {
		if (values.model)
			throw new ConfigurationError("--faux uses scripted test responses and cannot be mixed with a live model");
		if (values["faux-orientation"] !== "vertical" && values["faux-orientation"] !== "horizontal")
			throw new ConfigurationError("--faux-orientation must be vertical or horizontal");
		const models = createModels();
		const faux = fauxProvider({
			models: [
				{ id: "scripted-proposer", reasoning: false, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } },
			],
		});
		models.setProvider(faux.provider);
		const evidence = await controller.evidence();
		const state = await controller.waitForAcceptedWork();
		const adequate = state.evaluations.some((item) => item.result?.adequacy_gate_passed);
		faux.setResponses(
			adequate
				? [
						fauxAssistantMessage(
							fauxToolCall("finish_search", {
								outcome: "adequate",
								reason: "Scripted control: initial model passes the adequacy gate",
							}),
							{ stopReason: "toolUse" },
						),
					]
				: [
						fauxAssistantMessage(
							fauxToolCall("evaluate_hypothesis", {
								hypothesis: {
									kind: "band",
									orientation: values["faux-orientation"],
									center_fraction: 0.5,
									width_fraction: evidence.hypothesis_grammar.band_width_bounds[0],
								},
								reason: "Scripted test hypothesis, not a learned proposal",
							}),
							{ stopReason: "toolUse" },
						),
						fauxAssistantMessage(
							fauxToolCall("finish_search", {
								outcome: "adequate",
								reason: "Scripted finish after receiving validation diagnostics",
							}),
							{ stopReason: "toolUse" },
						),
					],
		);
		const result = await runProposer({
			controller,
			models,
			model: faux.getModel(),
			auditDirectory: values["audit-dir"],
			maxTurns: Number(values["max-turns"]),
			maxOutputTokens: Number(values["max-output-tokens"]),
		});
		console.log(JSON.stringify({ mode: "faux_scripted_smoke_test", campaign: result }, null, 2));
		return;
	}
	if (!values.model) throw new ConfigurationError("Live mode requires a Copilot --model; no scripted fallback is used");
	const copilotLogin = await resolveCopilotLogin();
	const models = createModels({ credentials: copilotLogin?.credentials });
	const copilot = githubCopilotProvider();
	models.setProvider(copilot);
	let model = models.getModel(copilot.id, values.model);
	if (!model)
		throw new ConfigurationError(
			`Copilot model ${values.model} is not in the installed Pi catalog; check model in the YAML config`,
		);
	if (copilotLogin?.kind === "github_oauth_direct") model = { ...model, baseUrl: copilotLogin.baseUrl };
	if (!(await models.getAuth(model)))
		throw new ConfigurationError("Sign in with opencode auth login github-copilot or set SCIENTIFIC_COPILOT_AUTH_FILE");
	const result = await runProposer({
		controller,
		models,
		model,
		auditDirectory: values["audit-dir"],
		maxTurns: Number(values["max-turns"]),
		maxOutputTokens: Number(values["max-output-tokens"]),
	});
	console.log(
		JSON.stringify({ mode: "live_llm", provider: model.provider, model: model.id, campaign: result }, null, 2),
	);
}

main().catch((error: unknown) => {
	if (error instanceof ConfigurationError || error instanceof CopilotLoginError) {
		console.error(error.message);
		process.exitCode = 1;
		return;
	}
	console.error(
		"Proposer failed. Check config, search/ledger.sqlite, and proposer/messages.jsonl; provider credentials are never printed.",
	);
	process.exitCode = 1;
});
