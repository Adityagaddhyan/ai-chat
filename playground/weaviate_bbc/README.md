# BBC News to Weaviate

This folder ingests the Kaggle dataset `gpreda/bbc-news` into a local Weaviate
instance.

The script:

1. Downloads the Kaggle dataset.
2. Loads the BBC News CSV.
3. Embeds `title + description` with `sentence-transformers`.
4. Creates a Weaviate class named `BBCNewsArticle`.
5. Batch inserts records with custom vectors.

## Prerequisites

Weaviate should be reachable at:

```bash
http://localhost:8080/v1/meta
```

Kaggle credentials must be configured in one of the standard ways:

```bash
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
```

or with `~/.kaggle/kaggle.json`.

Install the Kaggle CLI or Python package if you do not already have it:

```bash
uv add kaggle
```

Then authenticate:

```bash
uv run kaggle auth login
```

Alternatively, create an API token from your Kaggle account settings and put it
at `~/.kaggle/kaggle.json`, or export:

```bash
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
```

## Run

Smoke test with only 100 rows:

```bash
.venv/bin/python playground/weaviate_bbc/ingest_bbc_news.py --limit 100
```

Full ingest:

```bash
.venv/bin/python playground/weaviate_bbc/ingest_bbc_news.py
```

By default, the script also runs three example queries after ingestion:

- BM25 keyword search over `title` and `description`
- semantic vector search using the sentence-transformer query vector
- hybrid search combining BM25 and vector search with `alpha: 0.5`

Customize the example query:

```bash
.venv/bin/python playground/weaviate_bbc/ingest_bbc_news.py \
  --limit 100 \
  --query "latest climate and energy policy" \
  --query-limit 3
```

Skip the example queries:

```bash
.venv/bin/python playground/weaviate_bbc/ingest_bbc_news.py --skip-queries
```

Use an already downloaded CSV:

```bash
.venv/bin/python playground/weaviate_bbc/ingest_bbc_news.py \
  --csv-path playground/weaviate_bbc/data/bbc_news.csv \
  --skip-download
```

The script accepts either `http://localhost:8080` or
`http://localhost:8080/v1/meta` as the Weaviate URL.
