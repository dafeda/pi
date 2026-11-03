// Calls the loopback controller API from the Pi host.
import {
	checkEvaluation,
	checkEvidence,
	checkSnapshot,
	type Evaluation,
	type Evidence,
	type Hypothesis,
	type Snapshot,
} from "./contracts.ts";

/** Sends validated hypothesis requests to the Python controller. */
export class Controller {
	readonly url: URL;

	constructor(url: string) {
		this.url = new URL(url);
		if (
			this.url.protocol !== "http:" ||
			this.url.hostname !== "127.0.0.1" ||
			this.url.pathname !== "/" ||
			this.url.search ||
			this.url.hash ||
			this.url.username ||
			this.url.password
		) {
			throw new Error("Controller must use http://127.0.0.1 with a root path and no query or credentials");
		}
	}

	async request(path: string, body?: unknown, signal?: AbortSignal): Promise<unknown> {
		const response = await fetch(new URL(path, this.url), {
			method: body === undefined ? "GET" : "POST",
			headers: { "Content-Type": "application/json", Connection: "close" },
			body: body === undefined ? undefined : JSON.stringify(body),
			signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(45000)]) : AbortSignal.timeout(45000),
			redirect: "error",
		});
		const value: unknown = await response.json();
		if (!response.ok) {
			const error = value && typeof value === "object" && "error" in value ? String(value.error) : "request rejected";
			throw new Error(`Controller ${response.status}: ${error}`);
		}
		return value;
	}

	async evidence(): Promise<Evidence> {
		const value = await this.request("/v1/evidence");
		if (!checkEvidence.check(value)) throw new Error("Invalid public-evidence response schema");
		return value;
	}

	async snapshot(): Promise<Snapshot> {
		const value = await this.request("/v1/campaign");
		if (!checkSnapshot.check(value)) throw new Error("Invalid campaign response schema");
		return value;
	}

	async evaluate(hypothesis: Hypothesis, reason: string, signal?: AbortSignal): Promise<Evaluation> {
		const value = await this.request("/v1/evaluations", { hypothesis, reason }, signal);
		if (!checkEvaluation.check(value)) throw new Error("Invalid evaluation response schema");
		return value;
	}

	async finish(
		outcome: "adequate" | "inadequate" | "uncertain",
		reason: string,
		signal?: AbortSignal,
	): Promise<Snapshot> {
		const value = await this.request("/v1/finish", { outcome, reason }, signal);
		if (!checkSnapshot.check(value)) throw new Error("Invalid frozen-selection response schema");
		return value;
	}

	async waitForAcceptedWork(): Promise<Snapshot> {
		let snapshot = await this.snapshot();
		while (true) {
			const pending = snapshot.evaluations.find((item) => item.status === "queued" || item.status === "running");
			if (!pending) return snapshot;
			// Wait in the host so the next model request sees completed results, without LLM polling.
			const value = await this.request(`/v1/runs/${pending.run_id}/wait`);
			if (!checkEvaluation.check(value)) throw new Error("Invalid run-completion response schema");
			snapshot = await this.snapshot();
		}
	}
}
