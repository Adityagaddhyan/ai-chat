import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from embed import embed_documents, embed_query
from model import getModel
from retreiver_evals import evaluate_retrieval
from retriever import index_bm25, retrieve_top_k, searchBM25


TOP_K = 10
PREVIEW_LENGTH = 160
EMBEDDINGS_PATH = "embeddings/evals_doc_embeddings.npy"
MODEL = getModel()

TEST_QUERIES = [
    {
        "query": "advancements in space exploration technology",
        "desired_category": "sci.space",
    },
    {
        "query": "real-time rendering techniques in computer graphics",
        "desired_category": "comp.graphics",
    },
    {
        "query": "latest findings in cardiovascular medical research",
        "desired_category": "sci.med",
    },
    {
        "query": "NHL playoffs and team performance statistics",
        "desired_category": "rec.sport.hockey",
    },
    {
        "query": "impacts of cryptography in online security",
        "desired_category": "sci.crypt",
    },
    {
        "query": "the role of electronics in modern computing devices",
        "desired_category": "sci.electronics",
    },
    {
        "query": "motorcycles maintenance tips for enthusiasts",
        "desired_category": "rec.motorcycles",
    },
    {
        "query": "high-performance baseball tactics for championships",
        "desired_category": "rec.sport.baseball",
    },
    {
        "query": "historical influence of politics on society",
        "desired_category": "talk.politics.misc",
    },
    {
        "query": "latest technology trends in the Windows operating system",
        "desired_category": "comp.os.ms-windows.misc",
    },
]


def testSimilarity():
    documents = [
        "Mt. Fuji is a breathtaking place to explore during autumn.",
        "Santorini offers stunning views to admire during spring.",
        "Banff National Park is a picturesque destination to visit in the summer.",
        "The Great Wall of China is a spectacular site to experience during winter.",
        "The fjords of Norway are a magical place to cruise through in the spring.",
        "Prague is an enchanting city to wander through in winter.",
        "Kyoto's cherry blossoms create a beautiful scene to witness during spring.",
        "Marrakech offers vibrant markets and culture to enjoy in the fall.",
        "The Maldives are a paradisiacal getaway to savor during summer.",
        "The Christmas markets in Vienna are a festive delight to explore in winter.",
    ]
    query = "Suggest to me great places to visit in Asia."

    doc_embeddings = embed_documents(documents, MODEL)
    query_embedding = embed_query(query, MODEL)
    bm25_retriever = index_bm25(documents)

    cosine_docs = retrieve_top_k(
        query_embedding,
        doc_embeddings,
        documents,
        k=3,
        metric="cosine",
    )
    euclidean_docs = retrieve_top_k(
        query_embedding,
        doc_embeddings,
        documents,
        k=3,
        metric="euclidean",
    )
    bm25_docs, bm25_scores = searchBM25(query, bm25_retriever, top_k=3)

    print_text_results("Top 3 documents based on Cosine Similarity", cosine_docs)
    print_text_results("Top 3 documents based on Euclidean Distance", euclidean_docs)
    print_scored_text_results("Top 3 documents based on BM25", bm25_docs, bm25_scores)


def load_news_dataset():
    data = fetch_20newsgroups(subset="train")
    df = pd.DataFrame(
        {
            "text": data.data,
            "category": [data.target_names[target] for target in data.target],
        }
    )
    documents = df.to_dict("records")
    return df, documents


def load_or_create_embeddings(documents):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    if os.path.exists(EMBEDDINGS_PATH):
        return np.load(EMBEDDINGS_PATH)

    texts = [doc["text"] for doc in documents]
    doc_embeddings = embed_documents(texts, MODEL)
    np.save(EMBEDDINGS_PATH, doc_embeddings)
    return doc_embeddings


def retrieve_with_embeddings(query, doc_embeddings, documents, k):
    query_embedding = embed_query(query, MODEL)
    return retrieve_top_k(
        query_embedding,
        doc_embeddings,
        documents,
        k=k,
        metric="cosine",
    )


def evaluate_docs(retrieved_docs, desired_category, total_relevant_count):
    return evaluate_retrieval(
        retrieved_docs,
        desired_category,
        total_relevant_count,
    )


def evaluate_query(test_query, df, documents, doc_embeddings, bm25_retriever, k):
    query = test_query["query"]
    desired_category = test_query["desired_category"]
    total_relevant_count = count_category_docs(df, desired_category)

    embedding_docs = retrieve_with_embeddings(query, doc_embeddings, documents, k)
    bm25_docs, bm25_scores = searchBM25(query, bm25_retriever, top_k=k)

    return {
        "query": query,
        "desired_category": desired_category,
        "total_relevant_count": total_relevant_count,
        "embedding": {
            "docs": embedding_docs,
            "metrics": evaluate_docs(
                embedding_docs,
                desired_category,
                total_relevant_count,
            ),
        },
        "bm25": {
            "docs": bm25_docs,
            "scores": bm25_scores,
            "metrics": evaluate_docs(
                bm25_docs,
                desired_category,
                total_relevant_count,
            ),
        },
    }


def count_category_docs(df, category):
    return int((df["category"] == category).sum())


def collect_metrics(query_reports, retriever_name):
    return [report[retriever_name]["metrics"] for report in query_reports]


def print_text_results(title, docs):
    print(f"\n{title}:")
    for doc in docs:
        print(doc)


def print_scored_text_results(title, docs, scores):
    print(f"\n{title}:")
    for doc, score in zip(docs, scores):
        print(f"{score:.3f} {doc}")


def print_metrics(label, metrics, k):
    print(f"\n{label} metrics")
    print(f"Relevance@{k}: {metrics['relevance_matrix']}")
    print(f"Precision@{k}: {metrics['precision_at_k']:.3f}")
    print(f"Recall@{k}: {metrics['recall_at_k']:.3f}")
    print(f"Average Precision@{k}: {metrics['average_precision_at_k']:.3f}")
    print(f"Reciprocal Rank: {metrics['reciprocal_rank']:.3f}")


def print_top_results(label, retrieved_docs, k=3):
    print(f"{label} top results:")
    for index, doc in enumerate(retrieved_docs[:k], start=1):
        preview = " ".join(doc["text"].split())[:PREVIEW_LENGTH]
        print(f"{index}. [{doc['category']}] {preview}")


def print_aggregate_metrics(label, results, k):
    print(f"\n{label} aggregate metrics")
    print(f"Mean Precision@{k}: {mean_metric(results, 'precision_at_k'):.3f}")
    print(f"Mean Recall@{k}: {mean_metric(results, 'recall_at_k'):.3f}")
    print(
        f"Mean Average Precision@{k}: "
        f"{mean_metric(results, 'average_precision_at_k'):.3f}"
    )
    print(f"Mean Reciprocal Rank: {mean_metric(results, 'reciprocal_rank'):.3f}")


def mean_metric(results, metric_name):
    return np.mean([result[metric_name] for result in results])


def print_query_report(report, k):
    print(f"\nQuery: {report['query']}")
    print(f"Target category: {report['desired_category']}")
    print(f"Total relevant documents in dataset: {report['total_relevant_count']}")

    print_retriever_report("Embedding cosine", report["embedding"], k)
    print_retriever_report("BM25", report["bm25"], k)
    print(f"BM25 top scores: {format_top_scores(report['bm25']['scores'])}")


def print_retriever_report(label, result, k):
    print_metrics(label, result["metrics"], k)
    print_top_results(label, result["docs"])


def format_top_scores(scores, k=3):
    return [round(score, 3) for score in scores[:k]]


def run_evaluations(k=TOP_K):
    df, documents = load_news_dataset()
    doc_embeddings = load_or_create_embeddings(documents)
    bm25_retriever = index_bm25(documents)

    query_reports = [
        evaluate_query(
            test_query,
            df,
            documents,
            doc_embeddings,
            bm25_retriever,
            k,
        )
        for test_query in TEST_QUERIES
    ]

    for report in query_reports:
        print_query_report(report, k)

    print_aggregate_metrics(
        "Embedding cosine",
        collect_metrics(query_reports, "embedding"),
        k,
    )
    print_aggregate_metrics("BM25", collect_metrics(query_reports, "bm25"), k)


def evals():
    run_evaluations()


if __name__ == "__main__":
    evals()
