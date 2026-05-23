from scorer import get_cosiene_similarity, get_euclidean_distance, get_euclidean_distance
import numpy as np


def retrieve_top_k(query_embedding, doc_embeddings, documents, k=2, metric='cosine'):
    if metric == 'cosine':
        similarities = get_cosiene_similarity(query_embedding, doc_embeddings)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
    elif metric == 'euclidean':
        similarities = get_euclidean_distance(query_embedding, doc_embeddings)
        top_k_indices = np.argsort(similarities)[:k]
    top_k_docs = [documents[i] for i in top_k_indices]
    return top_k_docs

