import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_DATASET = "gpreda/bbc-news"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_WEAVIATE_URL = "http://localhost:8080"
DEFAULT_CLASS_NAME = "BBCNewsArticle"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_QUERY = "latest technology and business news"


def main():
    args = parse_args()
    weaviate_url = normalize_weaviate_url(args.weaviate_url)

    assert_weaviate_is_ready(weaviate_url)

    csv_path = resolve_csv_path(args)
    df = load_bbc_news(csv_path, limit=args.limit)
    model = SentenceTransformer(args.model_name)

    create_class_if_needed(weaviate_url, args.class_name)
    ingest_dataframe(
        df=df,
        model=model,
        weaviate_url=weaviate_url,
        class_name=args.class_name,
        batch_size=args.batch_size,
    )

    print(f"Ingested {len(df)} rows into Weaviate class {args.class_name}.")

    if not args.skip_queries:
        run_example_queries(
            query=args.query,
            model=model,
            weaviate_url=weaviate_url,
            class_name=args.class_name,
            limit=args.query_limit,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the Kaggle BBC News dataset and ingest it into Weaviate."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--weaviate-url", default=DEFAULT_WEAVIATE_URL)
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--query-limit", type=int, default=5)
    parser.add_argument(
        "--skip-queries",
        action="store_true",
        help="Only ingest data; do not run BM25, semantic, and hybrid examples.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing CSV in --data-dir or --csv-path.",
    )
    return parser.parse_args()


def normalize_weaviate_url(url):
    url = url.rstrip("/")

    if url.endswith("/v1/meta"):
        return url.removesuffix("/v1/meta")

    if url.endswith("/v1"):
        return url.removesuffix("/v1")

    return url


def assert_weaviate_is_ready(weaviate_url):
    meta = request_json("GET", f"{weaviate_url}/v1/meta")
    version = meta.get("version", "unknown")
    print(f"Connected to Weaviate {version} at {weaviate_url}.")


def resolve_csv_path(args):
    if args.csv_path:
        return require_file(args.csv_path)

    args.data_dir.mkdir(parents=True, exist_ok=True)

    existing_csv = find_csv(args.data_dir)
    if existing_csv and args.skip_download:
        return existing_csv

    if not args.skip_download and not existing_csv:
        download_kaggle_dataset(args.dataset, args.data_dir)

    csv_path = find_csv(args.data_dir)
    if not csv_path:
        raise FileNotFoundError(
            f"No CSV found in {args.data_dir}. "
            "Pass --csv-path or configure Kaggle credentials and rerun."
        )

    return csv_path


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def find_csv(data_dir):
    csv_files = sorted(data_dir.glob("*.csv"))
    return csv_files[0] if csv_files else None


def download_kaggle_dataset(dataset, data_dir):
    print(f"Downloading Kaggle dataset {dataset} into {data_dir}.")

    kaggle_package_error = try_download_with_kaggle_package(dataset, data_dir)
    if kaggle_package_error is None:
        return

    kaggle_cli_error = try_download_with_kaggle_cli(dataset, data_dir)
    if kaggle_cli_error is None:
        return

    errors = [
        error
        for error in [kaggle_package_error, kaggle_cli_error]
        if error is not False
    ]
    detail = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(
        "Could not download from Kaggle. Install/configure either the kaggle "
        "Python package or the kaggle CLI, then set KAGGLE_USERNAME and "
        "KAGGLE_KEY or place kaggle.json in ~/.kaggle/."
        + (f"\nDownload attempts failed with:\n{detail}" if detail else "")
    )


def try_download_with_kaggle_package(dataset, data_dir):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        return False

    if not has_kaggle_credentials():
        return (
            "kaggle Python package: missing credentials. Run `kaggle auth login`, "
            "set KAGGLE_USERNAME/KAGGLE_KEY, or create ~/.kaggle/kaggle.json."
        )

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=data_dir, unzip=True)
        unzip_downloads(data_dir)
        return None
    except Exception as error:
        return f"kaggle Python package: {error}"


def try_download_with_kaggle_cli(dataset, data_dir):
    if not shutil.which("kaggle"):
        return False

    if not has_kaggle_credentials():
        return (
            "kaggle CLI: missing credentials. Run `kaggle auth login`, "
            "set KAGGLE_USERNAME/KAGGLE_KEY, or create ~/.kaggle/kaggle.json."
        )

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(data_dir),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
        unzip_downloads(data_dir)
        return None
    except subprocess.CalledProcessError as error:
        return f"kaggle CLI: exited with status {error.returncode}"


def has_kaggle_credentials():
    has_env_credentials = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    has_config_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_env_credentials or has_config_file


def unzip_downloads(data_dir):
    for zip_path in data_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_dir)


def load_bbc_news(csv_path, limit=None):
    print(f"Loading {csv_path}.")
    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]

    required_columns = {"title", "description"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["title", "description"]).copy()
    df.drop_duplicates(subset=["title", "description"], inplace=True)
    if limit:
        df = df.head(limit)

    return df.reset_index(drop=True)


def create_class_if_needed(weaviate_url, class_name):
    if class_exists(weaviate_url, class_name):
        print(f"Weaviate class {class_name} already exists.")
        return

    schema = {
        "class": class_name,
        "description": "BBC News articles from the Kaggle gpreda/bbc-news dataset.",
        "vectorizer": "none",
        "properties": [
            {"name": "title", "dataType": ["text"]},
            {"name": "description", "dataType": ["text"]},
            {"name": "url", "dataType": ["text"]},
            {"name": "guid", "dataType": ["text"]},
            {"name": "published_at", "dataType": ["date"]},
            {"name": "source_dataset", "dataType": ["text"]},
        ],
    }
    request_json("POST", f"{weaviate_url}/v1/schema", body=schema)
    print(f"Created Weaviate class {class_name}.")


def class_exists(weaviate_url, class_name):
    try:
        request_json("GET", f"{weaviate_url}/v1/schema/{class_name}")
        return True
    except HTTPError as error:
        if error.code == 404:
            return False
        raise


def ingest_dataframe(df, model, weaviate_url, class_name, batch_size):
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(df), batch_size), start=1):
        batch_df = df.iloc[start : start + batch_size]
        texts = [build_article_text(row) for _, row in batch_df.iterrows()]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()
        objects = [
            build_weaviate_object(row, vector, class_name)
            for (_, row), vector in zip(batch_df.iterrows(), vectors)
        ]

        response = request_json(
            "POST",
            f"{weaviate_url}/v1/batch/objects",
            body={"objects": objects},
        )
        raise_for_batch_errors(response)
        print(f"Inserted batch {batch_number}/{total_batches} ({len(objects)} rows).")


def run_example_queries(query, model, weaviate_url, class_name, limit):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    print(f"\nExample query: {query}")
    print_query_results(
        "BM25 keyword search",
        query_bm25(weaviate_url, class_name, query, limit),
    )
    print_query_results(
        "Semantic vector search",
        query_semantic(weaviate_url, class_name, query_vector, limit),
    )
    print_query_results(
        "Hybrid search",
        query_hybrid(weaviate_url, class_name, query, query_vector, limit),
    )


def query_bm25(weaviate_url, class_name, query, limit):
    limit = validate_query_limit(limit)
    graphql_query = """
    query Search($query: String!) {
      Get {
        CLASS_NAME(
          bm25: {
            query: $query,
            properties: ["title", "description"]
          },
          limit: LIMIT
        ) {
          title
          description
          url
          _additional {
            score
          }
        }
      }
    }
    """.replace("CLASS_NAME", class_name).replace("LIMIT", str(limit))

    return run_graphql_query(
        weaviate_url,
        graphql_query,
        {"query": query},
        class_name,
    )


def query_semantic(weaviate_url, class_name, query_vector, limit):
    limit = validate_query_limit(limit)
    vector = format_graphql_vector(query_vector)
    graphql_query = """
    query Search {
      Get {
        CLASS_NAME(
          nearVector: {
            vector: VECTOR
          },
          limit: LIMIT
        ) {
          title
          description
          url
          _additional {
            distance
          }
        }
      }
    }
    """
    graphql_query = graphql_query.replace("CLASS_NAME", class_name)
    graphql_query = graphql_query.replace("LIMIT", str(limit))
    graphql_query = graphql_query.replace("VECTOR", vector)

    return run_graphql_query(
        weaviate_url,
        graphql_query,
        {},
        class_name,
    )


def query_hybrid(weaviate_url, class_name, query, query_vector, limit):
    limit = validate_query_limit(limit)
    vector = format_graphql_vector(query_vector)
    graphql_query = """
    query Search($query: String!) {
      Get {
        CLASS_NAME(
          hybrid: {
            query: $query,
            vector: VECTOR,
            alpha: 0.5,
            properties: ["title", "description"]
          },
          limit: LIMIT
        ) {
          title
          description
          url
          _additional {
            score
            explainScore
          }
        }
      }
    }
    """
    graphql_query = graphql_query.replace("CLASS_NAME", class_name)
    graphql_query = graphql_query.replace("LIMIT", str(limit))
    graphql_query = graphql_query.replace("VECTOR", vector)

    return run_graphql_query(
        weaviate_url,
        graphql_query,
        {"query": query},
        class_name,
    )


def format_graphql_vector(vector):
    return json.dumps([float(value) for value in vector])


def validate_query_limit(limit):
    limit = int(limit)
    if limit < 1:
        raise ValueError("--query-limit must be at least 1")
    return limit


def run_graphql_query(weaviate_url, graphql_query, variables, class_name):
    response = request_json(
        "POST",
        f"{weaviate_url}/v1/graphql",
        body={"query": graphql_query, "variables": variables},
    )

    if response.get("errors"):
        raise RuntimeError(json.dumps(response["errors"], indent=2))

    return response["data"]["Get"][class_name]


def print_query_results(title, results):
    print(f"\n{title}")
    for index, result in enumerate(results, start=1):
        preview = " ".join(result["description"].split())[:180]
        metadata = result.get("_additional", {})
        score = metadata.get("score", metadata.get("distance", "n/a"))
        print(f"{index}. score={score} title={result['title']}")
        print(f"   {preview}")


def build_article_text(row):
    return f"{row['title']}\n\n{row['description']}"


def build_weaviate_object(row, vector, class_name):
    guid = string_or_empty(row.get("guid"))
    link = string_or_empty(row.get("link"))
    object_key = guid or link or build_article_text(row)

    properties = {
        "title": string_or_empty(row.get("title")),
        "description": string_or_empty(row.get("description")),
        "url": link,
        "guid": guid,
        "source_dataset": DEFAULT_DATASET,
    }

    published_at = parse_datetime(row.get("pubDate"))
    if published_at:
        properties["published_at"] = published_at

    return {
        "class": class_name,
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, object_key)),
        "properties": properties,
        "vector": vector,
    }


def parse_datetime(value):
    if pd.isna(value):
        return None

    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None

    return timestamp.isoformat().replace("+00:00", "Z")


def string_or_empty(value):
    if pd.isna(value):
        return ""
    return str(value)


def raise_for_batch_errors(response):
    for item in response:
        result = item.get("result", {})
        errors = result.get("errors")
        if errors:
            raise RuntimeError(json.dumps(errors, indent=2))


def request_json(method, url, body=None):
    data = None
    headers = {"Content-Type": "application/json"}

    if body is not None:
        data = json.dumps(body).encode("utf-8")

    request = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    except HTTPError:
        raise
    except URLError as error:
        raise RuntimeError(f"Could not connect to {url}: {error}") from error

    if not payload:
        return None

    return json.loads(payload.decode("utf-8"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
