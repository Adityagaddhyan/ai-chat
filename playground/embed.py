def embed_documents(documents, model, batch_size=32):
    doc_embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
    return doc_embeddings

def embed_query(query, model):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    return query_embedding
