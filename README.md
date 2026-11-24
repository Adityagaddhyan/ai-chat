# AI Chat: RAG chatbot with tool calling, plus a retrieval lab

A hands-on project for building an AI support assistant end to end:

- **A chat app** (Node.js + Express + OpenAI) that answers from your documents with
  retrieval-augmented generation (RAG), calls tools to take actions, and streams replies.
- **A retrieval playground** (Python) for comparing retrieval methods and measuring them
  with real evaluation metrics.

<a href="https://github.com/sponsors/Adityagaddhyan"><img src="https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2?style=flat-square&logo=githubsponsors&logoColor=white" alt="Sponsor" /></a>
<a href="https://buymeacoffee.com/adityagaddhyan"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?style=flat-square&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee" /></a>

## The chat app

- **RAG over your docs:**
  - At startup, `docs/company-policy.txt` is chunked and embedded with OpenAI
    `text-embedding-3-small`.
  - Each question pulls the most relevant chunks into the prompt.
  - Retrieval combines cosine similarity with a keyword score.
- **Tool calling (agent loop):** the model can call typed tools defined with **Zod**, for
  example `getOrderStatus` and `cancelOrder`. The server runs them, feeds the results
  back, and loops until the model has a final answer.
- **Streaming:** replies stream to the browser over Server-Sent Events.
- **Conversation memory:** per-conversation history with a sliding window.

### Run it

Requires Node.js 20+ and an OpenAI API key.

```sh
git clone https://github.com/Adityagaddhyan/ai-chat.git
cd ai-chat
npm install
echo "OPENAI_API_KEY=sk-..." > .env
npm start
```

Open <http://localhost:3000>. To chat about your own content, replace
`docs/company-policy.txt`. The chat model is set in `server.js`.

## The retrieval playground

`playground/` is a set of Python experiments for understanding retrieval quality:

- **Dense retrieval** with `sentence-transformers` embeddings, using cosine or Euclidean
  distance
- **Sparse retrieval** with **BM25** (`bm25s`)
- **Evaluation:** recall@k and precision@k on a labelled set (`retreiver_evals.py`)
- **Vector database:** ingest the Kaggle BBC News dataset into **Weaviate** with custom
  vectors (see [`playground/weaviate_bbc`](playground/weaviate_bbc/README.md))

Run with [uv](https://docs.astral.sh/uv/) (Python 3.11+):

```sh
uv sync
uv run python playground/retreiver_evals.py
```

## Project layout

```
server.js          Express server: chat endpoint, agent loop, tools, SSE streaming
ragStore.js        Chunking, OpenAI embeddings, hybrid similarity search
public/            Chat UI
docs/              Knowledge base the bot answers from
playground/        Python retrieval experiments and evals
embeddings/        Cached embeddings for the evals
```

## Support

If this helped you learn RAG or agents, you can
[sponsor me on GitHub](https://github.com/sponsors/Adityagaddhyan) or
[buy me a coffee](https://buymeacoffee.com/adityagaddhyan) ☕
