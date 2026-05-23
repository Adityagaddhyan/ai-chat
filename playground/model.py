
from pathlib import Path
from sentence_transformers import SentenceTransformer

def getModel():
    model_path = Path.home() / "models" / "BAAI" / "bge-base-en-v1.5"
    model = SentenceTransformer(str(model_path))
    return model