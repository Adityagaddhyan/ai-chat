def recall_at_k(relevant_count, retrieved_relevant_count):
    if relevant_count < 0 or retrieved_relevant_count < 0:
        raise ValueError("Counts cannot be negative")
    if relevant_count == 0:
        return 0.0
    return retrieved_relevant_count / relevant_count


def precision_at_k(retrieved_relevant_count, total_count):
    if retrieved_relevant_count < 0 or total_count < 0:
        raise ValueError("Counts cannot be negative")
    if total_count == 0:
        return 0.0
    return retrieved_relevant_count / total_count


def build_relevance_matrix(retrieved_docs, desired_category):
    relevance_matrix = []
    for doc in retrieved_docs:
        if doc["category"] == desired_category:
            relevance_matrix.append(1)
        else:
            relevance_matrix.append(0)
    return relevance_matrix


def precision_matrix(relevance_matrix):
    relevant_count = 0
    matrix = []

    for index, relevant in enumerate(relevance_matrix):
        if relevant == 1:
            relevant_count += 1
        matrix.append(relevant_count / (index + 1))

    return matrix


def average_precision_at_k(relevance_matrix):
    precision_scores = precision_matrix(relevance_matrix)
    relevant_precision_scores = [
        precision for precision, relevant in zip(precision_scores, relevance_matrix)
        if relevant == 1
    ]

    if not relevant_precision_scores:
        return 0.0
    return sum(relevant_precision_scores) / len(relevant_precision_scores)


def reciprocal_rank(relevance_matrix):
    for index, relevant in enumerate(relevance_matrix):
        if relevant == 1:
            return 1 / (index + 1)
    return 0.0


def evaluate_retrieval(retrieved_docs, desired_category, total_relevant_count):
    relevance_matrix = build_relevance_matrix(retrieved_docs, desired_category)
    retrieved_relevant_count = sum(relevance_matrix)
    
    return {
        "precision_at_k": precision_at_k(retrieved_relevant_count, len(retrieved_docs)),
        "recall_at_k": recall_at_k(total_relevant_count, retrieved_relevant_count),
        "average_precision_at_k": average_precision_at_k(relevance_matrix),
        "reciprocal_rank": reciprocal_rank(relevance_matrix),
        "relevance_matrix": relevance_matrix,
        "retrieved_relevant_count": retrieved_relevant_count,
    }


# Backwards-compatible aliases for the earlier spellings.
prescision_at_k = precision_at_k
build_relevence_matrix = build_relevance_matrix
