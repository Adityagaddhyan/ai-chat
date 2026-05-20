import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import dotenv from 'dotenv';
import { z } from "zod";
import { zodFunction } from "openai/helpers/zod";
import { loadDocuments, searchRelevantChunks } from "./ragStore.js";
await loadDocuments();

const OrderStatusSchema = zodFunction({
    name: "getOrderStatus",
    description: "Get the status of an order by its ID",
    parameters: z.object({
        orderId: z.string().describe("The unique identifier of the order"),
    }),
});
const CancelOrderSchema = zodFunction({
    name: "cancelOrder",
    description: "Cancel an order by ID",
    parameters: z.object({
        orderId: z.string().describe("Order ID like ORD123 or ORD456"),
    }),
});

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const conversations = new Map();

const toolRegistry = {
    getOrderStatus: {
        definition: OrderStatusSchema,
        handler: getOrderStatus
    },
    cancelOrder: {
        definition: CancelOrderSchema,
        handler: cancelOrder,
    },
};

const MAX_TOOL_STEPS = 5;
const MAX_HISTORY_MESSAGES = 10;

app.post('/chat', async (req, res) => {
    const { message, conversationId = 'default' } = req.body;
    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    res.header('Content-Type', 'text/event-stream');
    res.header('Cache-Control', 'no-cache');
    res.header('Connection', 'keep-alive');

    const history = conversations.get(conversationId) || [];
    const currentUserMessage = {
        role: 'user',
        content: message
    };
    const systemPrompt = {
        role: 'system',
        content: `Use the retrieved context when relevant.
When answering from context, cite chunk IDs like [chunk-1].
If context is insufficient, say you do not know.`
    };
    try {
        const assistantText = await runAgentLoop(history, currentUserMessage, systemPrompt, res);

        history.push({ role: "user", content: message });
        history.push({ role: "assistant", content: assistantText });
        conversations.set(conversationId, history);

        res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
        res.end();

    } catch (error) {
        console.error('Error:', error);
        if (!res.headersSent) {
            return res.status(500).json({ error: 'An error occurred while processing your request.' });
        }

        res.write(`data: ${JSON.stringify({ error: 'An error occurred while processing your request.' })}\n\n`);
        res.end();
    }
});

async function runAgentLoop(history, currentUserMessage, systemPrompt, res) {
    const tools = Object.values(toolRegistry)
        .map(tool => tool.definition)
        .filter(Boolean);
    const relevantChunks = await searchRelevantChunks(
        currentUserMessage.content
    );
    console.log('Relevant chunks:', relevantChunks);
    const contextText = relevantChunks
        .map(c => `[${c.id} | ${c.source}]\n${c.text}`)
        .join("\n\n");
    const trimmedHistory = history.slice(-MAX_HISTORY_MESSAGES);
    systemPrompt.content += `\n\nRelevant information:\n${contextText}`;
    const conversationMessages = [systemPrompt, ...trimmedHistory, currentUserMessage];

    for (let step = 0; step < MAX_TOOL_STEPS; step++) {
        const result = await collectModelResponse(conversationMessages, tools, res);

        if (result.finishReason === "tool_calls") {
            if (result.toolCalls.length === 0) {
                throw new Error("Model finished with tool_calls but did not provide a tool call");
            }
            console.log('Processing tool calls:', JSON.stringify(result.toolCalls));

            conversationMessages.push({
                role: "assistant",
                content: null,
                tool_calls: result.toolCalls.map(formatToolCallMessage),
            });

            for (const toolCall of result.toolCalls) {
                const toolResult = await executeToolCall(toolCall);
                console.log('Tool call result:', toolResult);

                conversationMessages.push({
                    role: "tool",
                    tool_call_id: toolCall.id,
                    content: JSON.stringify(toolResult),
                });
            }

            continue;
        }

        if (result.finishReason === "stop") {
            return result.assistantText;
        }

        throw new Error(`Unexpected finish reason: ${result.finishReason}`);
    }

    throw new Error("Too many tool steps");
}

async function collectModelResponse(messages, tools, res = null) {

    const stream = await openai.chat.completions.create({
        model: "gpt-5.5",
        messages,
        tools,
        stream: true,
    });

    let assistantText = "";
    const toolCalls = new Map();

    for await (const event of stream) {
        // console.log('Received event:', JSON.stringify(event));
        const choice = event.choices?.[0];
        const delta = choice?.delta;
        const finishReason = choice?.finish_reason;

        if (delta?.content) {
            assistantText += delta.content;
            res?.write(`data: ${JSON.stringify({ delta: delta.content })}\n\n`);
        }

        for (const tc of delta?.tool_calls || []) {
            const index = tc.index ?? 0;

            if (!toolCalls.has(index)) {
                toolCalls.set(index, {
                    id: null,
                    name: null,
                    arguments: "",
                });
            }

            const current = toolCalls.get(index);

            if (tc.id) current.id = tc.id;
            if (tc.function?.name) current.name = tc.function.name;
            if (tc.function?.arguments) current.arguments += tc.function.arguments;

        }

        if (finishReason) {
            return {
                finishReason,
                assistantText,
                toolCalls: [...toolCalls.values()],
            };
        }
    }

    return {
        finishReason: "stop",
        assistantText,
        toolCalls: [...toolCalls.values()],
    };
}

function formatToolCallMessage(toolCall) {
    return {
        id: toolCall.id,
        type: "function",
        function: {
            name: toolCall.name,
            arguments: toolCall.arguments,
        },
    };
}

async function executeToolCall(toolCall) {
    const tool = toolRegistry[toolCall.name];

    if (!tool) {
        throw new Error(`Unknown tool: ${toolCall.name}`);
    }

    const args = JSON.parse(toolCall.arguments);
    return await tool.handler(args);
}

function getOrderStatus({ orderId }) {
    const orders = {
        ORD123: { status: "shipped", eta: "Tomorrow" },
        ORD456: { status: "processing", eta: "3 days" },
    };

    return orders[orderId] || { error: "Order not found" };
}



function cancelOrder({ orderId }) {
    return {
        orderId,
        cancelled: true,
        message: "Order cancelled successfully",
    };
}

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
