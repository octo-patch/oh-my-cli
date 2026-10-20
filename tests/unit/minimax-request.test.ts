import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import fs from "node:fs";
import path from "node:path";
// Use the SDK's supported web shim so only the HTTP boundary is mocked. The
// real SDK still serializes requests; no credentials or network are needed.
import "openai/shims/web";
import {
  parseProviderContract,
  resolveProviderConfig,
  selectProviderEntry,
} from "../../src/provider-contract.js";
import { invokeProvider } from "../../src/provider-invocation.js";
import { streamChat } from "../../src/provider.js";
import { runAgent } from "../../src/agent.js";
import { Workspace } from "../../src/workspace.js";
import type { SessionMessage } from "../../src/session.js";

const fetchMock = vi.hoisted(() => {
  const mock = vi.fn();
  vi.stubGlobal("fetch", mock);
  return mock;
});

const cases = ["minimax", "minimax-cn"].flatMap((id) => [
  { id, model: undefined, expectedModel: "MiniMax-M3.1-Flash", baseUrl: undefined },
  { id, model: "MiniMax-M3", expectedModel: "MiniMax-M3", baseUrl: undefined },
  { id, model: "MiniMax-M2.7", expectedModel: "MiniMax-M2.7", baseUrl: undefined },
  { id, model: "custom-model", expectedModel: "custom-model", baseUrl: "https://custom.example/v1" },
]);

function response(stream: boolean): Response {
  return new Response(stream
    ? 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    : JSON.stringify({ choices: [{ message: { content: "hello" }, finish_reason: "stop" }] }), {
    headers: { "content-type": stream ? "text/event-stream" : "application/json" },
  });
}

function capturedRequest() {
  expect(fetchMock).toHaveBeenCalledTimes(1);
  const [url, init] = fetchMock.mock.calls[0];
  expect(init.method).toBe("POST");
  return { url: String(url), body: JSON.parse(init.body as string) };
}

function expectedUrl(id: string, baseUrl?: string): string {
  return `${baseUrl ?? (id === "minimax" ? "https://api.minimax.io/v1" : "https://api.minimax.cn/v1")}/chat/completions`;
}

describe("MiniMax serialized requests", () => {
  let workspace: string;
  beforeAll(() => { workspace = fs.mkdtempSync(path.resolve(".minimax-request-")); });
  beforeEach(() => { fetchMock.mockReset(); });
  afterAll(() => {
    vi.unstubAllGlobals();
    fs.rmSync(workspace, { recursive: true, force: true });
  });

  it("retains streamed reasoning across a real agent tool round and resumed history", async () => {
    const contract = parseProviderContract({ contractVersion: 1, entries: [{ id: "minimax" }] });
    const config = resolveProviderConfig(selectProviderEntry(contract), { env: { MINIMAX_API_KEY: "test-key" } });
    const chunks = [
      { choices: [{ delta: { reasoning_content: "Inspect " } }] },
      { choices: [{ delta: { reasoning_content: "the file.", tool_calls: [{ index: 0, id: "call-read", type: "function", function: { name: "read", arguments: '{"path":"sample.txt"}' } }] }, finish_reason: "tool_calls" }] },
    ];
    fs.writeFileSync(path.join(workspace, "sample.txt"), "example");
    fetchMock.mockResolvedValueOnce(new Response(
      chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n",
      { headers: { "content-type": "text/event-stream" } },
    )).mockResolvedValueOnce(response(true));
    const persisted: SessionMessage[] = [];
    const assistantDelta = vi.fn();
    const result = await runAgent("Read sample.txt", [], {
      config,
      workspace: new Workspace(workspace),
      approvalMode: "yolo",
      sessionId: "minimax-test",
      onMessage: (message) => persisted.push(message),
      sink: {
        assistantDelta,
        assistantTurn: () => {}, toolStart: () => {}, toolResult: () => {},
        providerError: () => {}, usage: () => {}, retry: () => {},
      },
    });
    expect(result.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const assistant = persisted.find((message) => message.tool_calls);
    expect(assistant?.reasoning_content).toBe("Inspect the file.");
    const followup = JSON.parse(fetchMock.mock.calls[1][1].body as string);
    expect(followup.messages).toContainEqual(assistant);
    expect(followup.messages).toContainEqual(expect.objectContaining({ role: "tool", tool_call_id: "call-read" }));
    expect(assistantDelta.mock.calls).toEqual([["hello"]]);

    // JSON round-tripping models saved-session reload; the wire field survives.
    fetchMock.mockReset();
    fetchMock.mockResolvedValueOnce(response(true));
    for await (const _event of streamChat(config, JSON.parse(JSON.stringify(persisted)))) { /* drain */ }
    expect(capturedRequest().body.messages).toContainEqual(assistant);
  });

  it.each(cases)("invokes $id with $expectedModel through the default runner", async ({ id, model, expectedModel, baseUrl }) => {
    const entry = { id, ...(model ? { model } : {}), ...(baseUrl ? { baseUrl } : {}) };
    const settingsPath = path.join(workspace, "settings.json");
    fs.writeFileSync(settingsPath, JSON.stringify({ providers: { contractVersion: 1, entries: [entry] } }));
    fetchMock.mockResolvedValueOnce(response(false));

    const report = await invokeProvider({
      settingsPath,
      env: { MINIMAX_API_KEY: "test-key" },
      workspace,
      approvalMode: "yolo",
      prompt: "hello",
    });

    expect(report.outcome).toBe("called");
    expect(report.model).toBe(expectedModel);
    const request = capturedRequest();
    expect(request.url).toBe(expectedUrl(id, baseUrl));
    // Exact equality also guards against sending unsupported thinking, effort,
    // service tier or reasoning-split overrides with the default request.
    expect(request.body).toEqual({
      model: expectedModel,
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 256,
      stream: false,
    });
  });

  it.each(cases)("streams $id with $expectedModel and preserves tools and images", async ({ id, model, expectedModel, baseUrl }) => {
    const contract = parseProviderContract({
      contractVersion: 1,
      entries: [{ id, ...(model ? { model } : {}), ...(baseUrl ? { baseUrl } : {}) }],
    });
    const config = resolveProviderConfig(selectProviderEntry(contract), { env: { MINIMAX_API_KEY: "test-key" } });
    const toolCall = { id: "call-1", type: "function" as const, function: { name: "inspect", arguments: "{}" } };
    const image = "data:image/png;base64,aGVsbG8=";
    const messages: SessionMessage[] = [
      { role: "system", content: "Be helpful" },
      { role: "assistant", content: null, tool_calls: [toolCall] },
      { role: "tool", content: "done", tool_call_id: "call-1" },
      { role: "user", content: "Describe", images: [{ name: "test.png", mediaType: "image/png", bytes: 5, dataUrl: image }] },
    ];
    const tools = [{ type: "function" as const, function: { name: "inspect", description: "Inspect", parameters: { type: "object" } } }];
    fetchMock.mockResolvedValueOnce(response(true));

    const events = [];
    for await (const event of streamChat(config, messages, { tools })) events.push(event);

    expect(events).toEqual([{ type: "text", delta: "hello" }]);
    const request = capturedRequest();
    expect(request.url).toBe(expectedUrl(id, baseUrl));
    expect(request.body).toEqual({
      model: expectedModel,
      messages: [
        { role: "system", content: "Be helpful" },
        { role: "assistant", content: null, tool_calls: [toolCall] },
        { role: "tool", content: "done", tool_call_id: "call-1" },
        { role: "user", content: [{ type: "text", text: "Describe" }, { type: "image_url", image_url: { url: image } }] },
      ],
      stream: true,
      stream_options: { include_usage: true },
      tools,
    });
  });
});
