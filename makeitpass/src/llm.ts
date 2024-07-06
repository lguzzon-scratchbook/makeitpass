import assert from "node:assert";
import { OpenAI } from "openai";
import pc from "picocolors";
import { apiKey, maxTokens, model, prompt } from "./config";
import { log } from "./log";
import { spinner } from "./spinner";
import { executeTool, tools } from "./tools";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY
const OPENROUTER_BASE_URL =
  process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1"

const openai = new OpenAI({
  apiKey: OPENROUTER_API_KEY, // defaults to process.env["OPENAI_API_KEY"]
  baseURL: OPENROUTER_BASE_URL,
  defaultHeaders: {
    "HTTP-Referer": "https://github.com/lguzzon-scratchbook/makeitpass",
  },
  // dangerouslyAllowBrowser: true, // Enable this if you used OAuth to fetch a user-scoped `apiKey` above. See https://openrouter.ai/docs#oauth to learn how.
})
const client = openai

function composePrompt(opts: { command: string; stdout: string; error: string }): string {
  return `
The following command failed:
${opts.command}
  
stdout:
${opts.stdout}
  
stderr:
${opts.error}

${prompt}`;
}

const messageHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [];

export async function applyFix(opts: {
  command: string;
  stdout: string;
  error: string;
  iteration: number;
}): Promise<OpenAI.Chat.ChatCompletion.Choice> {
  try {
    const prompt = composePrompt(opts);
    messageHistory.push({
      role: "user",
      content: prompt,
    });

    while (true) {
      log("asking GPT...", 2);
      let response = await spinner(
        "GPT is thinking",
        client.chat.completions.create({
          model,
          max_tokens: maxTokens,
          messages: messageHistory,
          tools,
        }),
      );
      
      const assistantMessage = response.choices[0].message;
      messageHistory.push(assistantMessage);
      
      assert(assistantMessage.tool_calls, "expected tool calls in response");

      const textContent = assistantMessage.content;
      if (textContent) {
        console.log(pc.cyan(`${pc.yellow(`Iteration ${opts.iteration}`)}: 🤖 ${textContent}`));
      }
      
      const toolCalls = assistantMessage.tool_calls;
      assert(toolCalls.length > 0, `no tool found in response: ${JSON.stringify(response)}`);

      // run tools in parallel
      const toolResults = await spinner(
        "Running tools",
        Promise.all(
          toolCalls.map(async (toolCall) => {
            log(`GPT is using tool: ${toolCall.function.name}`, 2);
            const toolResult = await executeTool({
              name: toolCall.function.name,
              input: JSON.parse(toolCall.function.arguments),
              iteration: opts.iteration,
            });
            assert(typeof toolResult === "string", `tool result should be string: ${toolResult}`);
            log(`tool result: ${toolResult}`, 2);
            return {
              tool_call_id: toolCall.id,
              role: "tool" as const,
              content: toolResult,
            };
          }),
        ),
      );
      messageHistory.push(...toolResults);

      response = await spinner(
        "GPT is thinking",
        client.chat.completions.create({
          model,
          max_tokens: maxTokens,
          messages: messageHistory,
          tools,
        }),
      );

      // keep looping until decides not to use tools anymore
      if (response.choices[0].message.tool_calls) continue;
      return response.choices[0];
    }
  } catch (error) {
    console.dir(messageHistory, { depth: null });
    throw error;
  }
}
