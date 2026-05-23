import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from embed import embed_documents, embed_query
from model import getModel
from retreiver_evals import evaluate_retrieval
from retriever import retrieve_top_k


model = getModel()


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

    query_embeddings = embed_query(query, model)
    doc_embeddings = embed_documents(documents, model)
    retrieved_docs_cosine = retrieve_top_k(
        query_embeddings, doc_embeddings, documents, k=3, metric="cosine"
    )
    retrieved_docs_euclidean = retrieve_top_k(
        query_embeddings, doc_embeddings, documents, k=3, metric="euclidean"
    )
    print("Top 3 documents based on Cosine Similarity:")
    for doc in retrieved_docs_cosine:
        print(doc)
    print("\nTop 3 documents based on Euclidean Distance:")
    for doc in retrieved_docs_euclidean:
        print(doc)


def evals():
    k = 10
    test_queries = [
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

    data = fetch_20newsgroups(subset="train")
    df = pd.DataFrame(
        {
            "text": data.data,
            "category": [data.target_names[i] for i in data.target],
        }
    )

    documents = df.to_dict("records")
    embeddings_path = "embeddings/evals_doc_embeddings.npy"
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

    if not os.path.exists(embeddings_path):
        doc_embeddings = embed_documents(df["text"].tolist(), model)
        np.save(embeddings_path, doc_embeddings)
    else:
        doc_embeddings = np.load(embeddings_path)

    results = []

    for test_query in test_queries:
        query_embedding = embed_query(test_query["query"], model)
        retrieved_docs = retrieve_top_k(
            query_embedding,
            doc_embeddings,
            documents,
            k=k,
            metric="cosine",
        )
        desired_category = test_query["desired_category"]
        total_relevant_count = int((df["category"] == desired_category).sum())
        metrics = evaluate_retrieval(
            retrieved_docs,
            desired_category,
            total_relevant_count,
        )
        results.append(metrics)

        print(f"\nQuery: {test_query['query']}")
        print(f"Target category: {desired_category}")
        print(f"Relevance@{k}: {metrics['relevance_matrix']}")
        print(f"Precision@{k}: {metrics['precision_at_k']:.3f}")
        print(f"Recall@{k}: {metrics['recall_at_k']:.3f}")
        print(f"Average Precision@{k}: {metrics['average_precision_at_k']:.3f}")
        print(f"Reciprocal Rank: {metrics['reciprocal_rank']:.3f}")

        print("Top results:")
        for index, doc in enumerate(retrieved_docs[:3], start=1):
            preview = " ".join(doc["text"].split())[:160]
            print(f"{index}. [{doc['category']}] {preview}")

    print("\nAggregate metrics")
    print(f"Mean Precision@{k}: {np.mean([r['precision_at_k'] for r in results]):.3f}")
    print(f"Mean Recall@{k}: {np.mean([r['recall_at_k'] for r in results]):.3f}")
    print(
        f"Mean Average Precision@{k}: "
        f"{np.mean([r['average_precision_at_k'] for r in results]):.3f}"
    )
    print(f"Mean Reciprocal Rank: {np.mean([r['reciprocal_rank'] for r in results]):.3f}")
    


if __name__ == "__main__":
    evals()
