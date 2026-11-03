// Reuses saved Copilot OAuth authorization without persisting or exposing credentials.
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";
import { InMemoryCredentialStore, type OAuthCredential } from "@earendil-works/pi-ai";

export type CopilotLogin =
	| { kind: "copilot_token_exchange"; credentials: InMemoryCredentialStore }
	| { kind: "github_oauth_direct"; credentials: InMemoryCredentialStore; baseUrl: string };

/** A setup error whose message is safe to show without revealing credentials. */
export class CopilotLoginError extends Error {}

/** Uses the explicit login file or OpenCode's active Copilot account without interactive login. */
export async function resolveCopilotLogin(): Promise<CopilotLogin | undefined> {
	const path = process.env.SCIENTIFIC_COPILOT_AUTH_FILE;
	if (path) return loadCopilotLogin(path);
	let stdout: string;
	try {
		({ stdout } = await promisify(execFile)("opencode", ["auth", "export", "github-copilot"], {
			timeout: 30_000,
		}));
	} catch (error) {
		if (error instanceof Error && "code" in error && error.code === "ENOENT") return undefined;
		throw new CopilotLoginError(
			"Cannot export the saved Copilot login from OpenCode; check opencode auth login github-copilot",
		);
	}
	let data: unknown;
	try {
		data = JSON.parse(stdout);
	} catch {
		throw new CopilotLoginError("OpenCode returned an invalid Copilot credential export");
	}
	if (!Array.isArray(data)) {
		throw new CopilotLoginError("OpenCode returned an invalid Copilot credential export");
	}
	// The external CLI export is a credential boundary; ignore other providers and inactive accounts.
	const accounts: unknown[] = data;
	for (const account of accounts) {
		if (
			!account ||
			typeof account !== "object" ||
			Array.isArray(account) ||
			!("integrationID" in account) ||
			account.integrationID !== "github-copilot" ||
			!("active" in account) ||
			account.active !== true
		) {
			continue;
		}
		if (!("value" in account)) throw new CopilotLoginError("The active Copilot account has no credential");
		return createCopilotLogin(account.value);
	}
	return undefined;
}

/** Loads a saved login; uses its GitHub token directly or exchanges it for a Copilot access token. */
export async function loadCopilotLogin(path: string): Promise<CopilotLogin> {
	let data: unknown;
	try {
		data = JSON.parse(await readFile(path, "utf8"));
	} catch {
		throw new CopilotLoginError("Cannot read the saved Copilot login file");
	}
	if (!data || typeof data !== "object" || Array.isArray(data) || !("github-copilot" in data)) {
		throw new CopilotLoginError("The login file has no github-copilot credential");
	}
	return createCopilotLogin(data["github-copilot"]);
}

async function createCopilotLogin(entry: unknown): Promise<CopilotLogin> {
	// Validate credentials from the operator's auth source, never model-supplied input.
	if (
		!entry ||
		typeof entry !== "object" ||
		Array.isArray(entry) ||
		!("type" in entry) ||
		entry.type !== "oauth" ||
		!("refresh" in entry) ||
		typeof entry.refresh !== "string" ||
		entry.refresh.length === 0
	) {
		throw new CopilotLoginError(
			"Saved Copilot login requires GitHub OAuth authorization and a non-empty refresh token",
		);
	}
	const credential: OAuthCredential = { type: "oauth", refresh: entry.refresh, access: "", expires: 0 };
	if ("enterpriseUrl" in entry && entry.enterpriseUrl !== undefined) {
		if (typeof entry.enterpriseUrl !== "string") throw new CopilotLoginError("Copilot enterpriseUrl must be a string");
		credential.enterpriseUrl = entry.enterpriseUrl;
	}
	const credentials = new InMemoryCredentialStore();
	if ("access" in entry && entry.access === entry.refresh) {
		// OpenCode's GitHub token works directly; the service that issues Pi's Copilot tokens rejects it.
		await credentials.modify("github-copilot", async () => ({ type: "api_key", key: credential.refresh }));
		const enterprise = credential.enterpriseUrl;
		const domain =
			typeof enterprise === "string" && enterprise.length > 0
				? new URL(enterprise.includes("://") ? enterprise : `https://${enterprise}`).hostname
				: undefined;
		return {
			kind: "github_oauth_direct",
			credentials,
			baseUrl: domain ? `https://copilot-api.${domain}` : "https://api.githubcopilot.com",
		};
	}
	await credentials.modify("github-copilot", async () => credential);
	return { kind: "copilot_token_exchange", credentials };
}
