from scorer import (
    get_cosiene_similarity,
    get_euclidean_distance,
)
import numpy as np
import bm25s


def retrieve_top_k(query_embedding, doc_embeddings, documents, k=2, metric="cosine"):
    if metric == "cosine":
        similarities = get_cosiene_similarity(query_embedding, doc_embeddings)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
    elif metric == "euclidean":
        similarities = get_euclidean_distance(query_embedding, doc_embeddings)
        top_k_indices = np.argsort(similarities)[:k]
    else:
        raise ValueError(f"Unsupported retrieval metric: {metric}")

    return [documents[i] for i in top_k_indices]


def get_document_text(doc):
    if isinstance(doc, dict):
        return doc["text"]
    return doc


def index_bm25(docs: list):
    corpus_texts = [get_document_text(doc) for doc in docs]
    corpus_tokens = bm25s.tokenize(corpus_texts)
    retriever = bm25s.BM25(corpus=docs)
    retriever.index(corpus=corpus_tokens, show_progress=True)
    return retriever


def searchBM25(query: str, retriever: bm25s.BM25, top_k=5):
    query_token = bm25s.tokenize(query)
    docs, score = retriever.retrieve(query_token, k=top_k)
    return docs[0].tolist(), score[0].tolist()
