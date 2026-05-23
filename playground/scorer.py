import numpy as np


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def get_cosiene_similarity(query_embedding, doc_embeddings):
    similarities = []
    for doc_embedding in doc_embeddings:
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)
    return similarities


def get_euclidean_distance(query_embedding, doc_embeddings):
    distances = []
    for doc_embedding in doc_embeddings:
        distance = euclidean_distance(query_embedding, doc_embedding)
        distances.append(distance)
    return distances


