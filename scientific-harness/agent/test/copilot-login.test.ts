// Checks saved Copilot logins and renewed access tokens without real credentials or live model requests.
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import test, { type TestContext } from "node:test";
import { createModels, fauxProvider } from "@earendil-works/pi-ai";
import { githubCopilotProvider } from "@earendil-works/pi-ai/providers/github-copilot";
import { CopilotLoginError, loadCopilotLogin, resolveCopilotLogin } from "../src/copilot-login.ts";

test("loads only Copilot authorization and obtains fresh access without rewriting the login file", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-copilot-login-"));
	try {
		const path = join(directory, "auth.json");
		const source = JSON.stringify({
			"github-copilot": {
				type: "oauth",
				refresh: "fixture-github-authorization",
				access: "fixture-not-a-copilot-token",
				expires: Date.now() + 86400000,
			},
			openai: { type: "api", key: "unrelated-fixture-secret" },
		});
		await writeFile(path, source, { mode: 0o600 });
		const login = await loadCopilotLogin(path);
		assert.equal(login.kind, "copilot_token_exchange");
		const credentials = login.credentials;
		assert.deepEqual(await credentials.list(), [{ providerId: "github-copilot", type: "oauth" }]);
		assert.deepEqual(await credentials.read("github-copilot"), {
			type: "oauth",
			refresh: "fixture-github-authorization",
			access: "",
			expires: 0,
		});
		let refreshes = 0;
		const faux = fauxProvider();
		const models = createModels({ credentials });
		models.setProvider({
			...faux.provider,
			id: "github-copilot",
			auth: {
				oauth: {
					name: "Fixture Copilot",
					login: async () => {
						throw new Error("No interactive login is needed");
					},
					refresh: async (credential) => {
						refreshes += 1;
						assert.equal(credential.refresh, "fixture-github-authorization");
						return { ...credential, access: "fixture-copilot-token", expires: Date.now() + 86400000 };
					},
					toAuth: async (credential) => ({ apiKey: credential.access, baseUrl: "https://fixture.invalid" }),
				},
			},
		});
		const first = await models.getAuth("github-copilot");
		assert.equal(first?.auth.apiKey, "fixture-copilot-token");
		assert.equal(first?.auth.baseUrl, "https://fixture.invalid");
		assert.deepEqual(await models.getAuth("github-copilot"), first);
		assert.equal(refreshes, 1);
		assert.equal(await readFile(path, "utf8"), source);
		assert.equal(faux.state.callCount, 0);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("retains the enterprise domain when loading a Pi Copilot login", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-copilot-login-"));
	try {
		const path = join(directory, "auth.json");
		await writeFile(
			path,
			JSON.stringify({
				"github-copilot": {
					type: "oauth",
					refresh: "fixture-github-authorization",
					access: "fixture-copilot-token",
					expires: 0,
					enterpriseUrl: "company.ghe.com",
					availableModelIds: ["fixture-model"],
				},
			}),
			{ mode: 0o600 },
		);
		const { credentials } = await loadCopilotLogin(path);
		const loaded = await credentials.read("github-copilot");
		assert.ok(loaded?.type === "oauth");
		assert.equal(loaded.enterpriseUrl, "company.ghe.com");
		assert.equal(loaded.access, "");
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("an OpenCode GitHub-token login selects the direct endpoint without token exchange", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-copilot-login-"));
	try {
		const path = join(directory, "auth.json");
		const source = JSON.stringify({
			"github-copilot": {
				type: "oauth",
				refresh: "fixture-github-authorization",
				access: "fixture-github-authorization",
				expires: 0,
			},
		});
		await writeFile(path, source, { mode: 0o600 });
		const login = await loadCopilotLogin(path);
		assert.ok(login.kind === "github_oauth_direct");
		assert.equal(login.baseUrl, "https://api.githubcopilot.com");
		const models = createModels({ credentials: login.credentials });
		models.setProvider(githubCopilotProvider());
		const auth = await models.getAuth("github-copilot");
		assert.equal(auth?.auth.apiKey, "fixture-github-authorization");
		assert.equal(await readFile(path, "utf8"), source);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("rejects missing or malformed OAuth authorization without revealing file contents", async () => {
	const directory = await mkdtemp(join(tmpdir(), "scientific-copilot-login-"));
	try {
		const path = join(directory, "auth.json");
		for (const value of [
			{ "github-copilot": { type: "api_key", key: "fixture-secret" } },
			{ "github-copilot": { type: "oauth", access: "fixture-secret" } },
			{ "github-copilot": { type: "oauth", refresh: "" } },
			{ "github-copilot": { type: "oauth", refresh: 123 } },
			{ "github-copilot": { type: "oauth", refresh: "fixture-secret", enterpriseUrl: 123 } },
			{},
			null,
			[],
		]) {
			await writeFile(path, JSON.stringify(value), { mode: 0o600 });
			await assert.rejects(loadCopilotLogin(path), (error) => {
				assert.ok(error instanceof Error);
				assert.ok(!error.message.includes("fixture-secret"));
				return true;
			});
		}
		await writeFile(path, "{fixture-secret", { mode: 0o600 });
		await assert.rejects(loadCopilotLogin(path), /Cannot read the saved Copilot login file/);
		await assert.rejects(loadCopilotLogin(join(directory, "missing.json")), /Cannot read/);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("automatically reuses only the active OpenCode Copilot account", async (context) => {
	const accounts = [
		{
			integrationID: "github-copilot",
			active: false,
			value: { type: "oauth", refresh: "inactive-fixture-secret", access: "inactive-fixture-secret", expires: 0 },
		},
		{
			integrationID: "openai",
			active: true,
			value: { type: "oauth", refresh: "unrelated-fixture-secret", access: "unrelated-fixture-secret", expires: 0 },
		},
		{
			integrationID: "github-copilot",
			active: true,
			value: { type: "oauth", refresh: "active-fixture-secret", access: "active-fixture-secret", expires: 0 },
		},
	];
	await createOpenCodeFixture(context, `process.stdout.write(${JSON.stringify(JSON.stringify(accounts))});`);
	const login = await resolveCopilotLogin();
	assert.ok(login?.kind === "github_oauth_direct");
	assert.equal(login.baseUrl, "https://api.githubcopilot.com");
	assert.deepEqual(await login.credentials.list(), [{ providerId: "github-copilot", type: "api_key" }]);
	assert.deepEqual(await login.credentials.read("github-copilot"), {
		type: "api_key",
		key: "active-fixture-secret",
	});
});

test("an explicit login file takes precedence over automatic OpenCode discovery", async (context) => {
	const directory = await createOpenCodeFixture(context, "process.exit(1);");
	const path = join(directory, "auth.json");
	const source = JSON.stringify({
		"github-copilot": { type: "oauth", refresh: "fixture-github-authorization", access: "expired", expires: 0 },
	});
	await writeFile(path, source, { mode: 0o600 });
	process.env.SCIENTIFIC_COPILOT_AUTH_FILE = path;
	const login = await resolveCopilotLogin();
	assert.ok(login?.kind === "copilot_token_exchange");
	assert.equal(await readFile(path, "utf8"), source);
});

test("an invalid explicit login file never falls back to another account", async (context) => {
	const directory = await createOpenCodeFixture(context, "process.stdout.write('[]');");
	const path = join(directory, "auth.json");
	await writeFile(path, "{}", { mode: 0o600 });
	process.env.SCIENTIFIC_COPILOT_AUTH_FILE = path;
	await assert.rejects(resolveCopilotLogin(), /no github-copilot credential/);
});

test("missing OpenCode leaves ambient provider credentials available", async (context) => {
	const directory = await createOpenCodeFixture(context, "process.stdout.write('[]');");
	await rm(join(directory, "opencode"));
	process.env.PATH = directory;
	assert.equal(await resolveCopilotLogin(), undefined);
});

test("does not select an inactive account when no Copilot login is active", async (context) => {
	const accounts = [
		{
			integrationID: "github-copilot",
			active: false,
			value: { type: "oauth", refresh: "inactive-fixture-secret", access: "inactive-fixture-secret", expires: 0 },
		},
	];
	await createOpenCodeFixture(context, `process.stdout.write(${JSON.stringify(JSON.stringify(accounts))});`);
	assert.equal(await resolveCopilotLogin(), undefined);
});

test("OpenCode export failures never expose credential-bearing process output", async (context) => {
	await createOpenCodeFixture(
		context,
		"process.stdout.write('fixture-secret'); process.stderr.write('fixture-secret'); process.exit(1);",
	);
	await assert.rejects(resolveCopilotLogin(), (error: unknown) => {
		assert.ok(error instanceof CopilotLoginError);
		assert.ok(!error.message.includes("fixture-secret"));
		assert.ok(!("stdout" in error) && !("stderr" in error) && !("cause" in error));
		return true;
	});
});

test("rejects malformed OpenCode exports without exposing credential contents", async (context) => {
	const account = { integrationID: "github-copilot", active: true };
	const cases = [
		["invalid JSON", "{fixture-secret"],
		["invalid export shape", JSON.stringify({ secret: "fixture-secret" })],
		["missing credential", JSON.stringify([account])],
		[
			"invalid credential",
			JSON.stringify([{ ...account, value: { type: "oauth", refresh: 123, access: "fixture-secret" } }]),
		],
	];
	for (const [name, output] of cases) {
		await context.test(name, async (child) => {
			await createOpenCodeFixture(child, `process.stdout.write(${JSON.stringify(output)});`);
			await assert.rejects(resolveCopilotLogin(), (error: unknown) => {
				assert.ok(error instanceof CopilotLoginError);
				assert.ok(!error.message.includes("fixture-secret"));
				return true;
			});
		});
	}
});

async function createOpenCodeFixture(context: TestContext, body: string): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "scientific-opencode-login-"));
	context.after(() => rm(directory, { recursive: true, force: true }));
	const previous = {
		PATH: process.env.PATH,
		SCIENTIFIC_COPILOT_AUTH_FILE: process.env.SCIENTIFIC_COPILOT_AUTH_FILE,
	};
	context.after(() => {
		for (const [name, value] of Object.entries(previous)) {
			if (value === undefined) delete process.env[name];
			else process.env[name] = value;
		}
	});
	// An executable fixture prevents automatic discovery from touching real accounts or provider APIs.
	await writeFile(
		join(directory, "opencode"),
		"#!/usr/bin/env node\n" +
			`if (JSON.stringify(process.argv.slice(2)) !== '["auth","export","github-copilot"]') process.exit(2);\n` +
			body,
		{ mode: 0o700 },
	);
	process.env.PATH = `${directory}${delimiter}${previous.PATH ?? ""}`;
	delete process.env.SCIENTIFIC_COPILOT_AUTH_FILE;
	return directory;
}
