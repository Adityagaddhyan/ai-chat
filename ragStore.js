import fs from "fs/promises";
import OpenAI from "openai";
import dotenv from "dotenv";
import { id } from "zod/locales";
dotenv.config();

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const chunks = [];

function chunkText(text) {
    return text
        .split("\n\n")
        .map(chunk => chunk.trim())
        .filter(Boolean);
}

async function createEmbedding(text) {
    const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: text,
    });

    return response.data[0].embedding;
}

function cosineSimilarity(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b)) {
        throw new Error("cosineSimilarity expected two embedding arrays");
    }
    if (a.length !== b.length) {
        throw new Error("cosineSimilarity expected embeddings with the same length");
    }

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function loadDocuments() {
    const text = await fs.readFile(
        "./docs/company-policy.txt",
        "utf-8"
    );

    const splitChunks = chunkText(text);
    let i = 1;
    for (const chunk of splitChunks) {
        const embedding = await createEmbedding(chunk);
        // console.log(`Created embedding for chunk: ${chunk}`);
        // console.log(`Embedding: ${embedding}`);
        chunks.push({
            text: chunk,
            embedding,
            id: `chunk-${i++}`,
            source: "company-policy.txt",
        });
    }
    // console.log(chunks);
    console.log(`Loaded ${chunks.length} chunks`);
}

export async function searchRelevantChunks(query, topK = 1, scoreThreshold = 0.1) {
    const queryEmbedding = await createEmbedding(query);

    const scored = chunks.map(chunk => {
        const semanticScore = cosineSimilarity(
            queryEmbedding,
            chunk.embedding
        );
        const lexicalScore = keywordScore(query, chunk.text);
        const combinedScore =
            (semanticScore * 0.7) +
            (lexicalScore * 0.3);

        return {
            text: chunk.text,
            score: combinedScore,
            source: chunk.source,
            id: chunk.id,
        };
    });

    scored.sort((a, b) => b.score - a.score);

    return scored.filter(c => c.score >= scoreThreshold).slice(0, topK)
        .map(c => ({
            text: c.text,
            source: c.source,
            id: c.id,
            score: c.score,
        }));
}

export function keywordScore(query, text) {
    const queryTerms = query
        .toLowerCase()
        .split(/\s+/);

    const textLower = text.toLowerCase();

    let score = 0;
    for (const term of queryTerms) {
        if (textLower.includes(term)) {
            score += 1;
        }
    }

    return score;
}
