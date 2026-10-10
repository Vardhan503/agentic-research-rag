# Agentic Research RAG

Question answering over 4,075 open-access papers on retrieval-augmented
generation, information retrieval, LLM agents and hallucination detection.
The system is a LangGraph graph that routes each question, retrieves with a
hybrid dense + sparse index, grades the evidence (CRAG), rewrites the query or
falls back to web search when the corpus is not enough, and checks its own
answer for grounding and completeness (Self-RAG) before returning it with
citations.

Four pipelines of increasing capability are evaluated on the same 71-question
benchmark so that each added component has to justify its latency.

## How it works

```
question
  └─ route_question ─── direct answer (no retrieval needed)
       │
       └─ retrieve (FAISS bge-small + SQLite FTS5 BM25 → RRF → cross-encoder rerank)
            └─ grade_documents (CRAG: per-chunk relevance, then combined-context check)
                 ├─ correct   ─────────────────────────┐
                 ├─ ambiguous → rewrite_query → retrieve │  (bounded: 1 rewrite)
                 └─ incorrect → web_search (Tavily, date-filtered for time-bound questions)
                                                        ▼
                                                 generate_answer (cited)
                                                        │
                                          check_hallucination → critique_answer
                                            (Self-RAG, bounded retries)
                                                        │
                                        answer | insufficient_evidence | needs_more_context
```

Key modules live under `src/agentic_rag/`:

| Path | Role |
| --- | --- |
| `ingestion/`, `processing/` | OpenAlex discovery, PDF/JATS/TEI parsing, screening, chunking |
| `retrieval/hybrid.py` | Hybrid retriever (dense + sparse, RRF, reranking, per-paper caps) |
| `graph/workflow.py` | LangGraph `StateGraph` wiring and routing logic |
| `graph/crag.py`, `graph/self_rag.py` | Evidence grading and answer self-checks |
| `graph/web_search.py` | Tavily fallback with publication-date filtering |
| `evaluation/` | Benchmark loader, deterministic metrics, RAGAS judge, pooled gold labelling |

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" -r requirements-evaluation.txt

cp .env.example .env   # then fill in OPENAI_API_KEY (and TAVILY_API_KEY for web fallback)
```

Model choices are in `configs/corpus.yaml` under `agentic_rag`. The default
provider is OpenAI `gpt-4.1-mini`; set `provider: ollama` to run fully local.

## Build the index

The processed corpus is not committed. To rebuild it from scratch run the
ingestion scripts in order (each is resumable and writes a report under
`data/interim/`):

```bash
python scripts/discover_corpus.py          # OpenAlex metadata by topic
python scripts/filter_corpus.py            # rule-based screening
python scripts/screen_ambiguous_corpus.py  # LLM screening of borderline papers
python scripts/enrich_oa_locations.py      # resolve open-access full-text URLs
python scripts/download_direct_oa_pdfs.py  # PDFs (publisher, MDPI CDN, OpenAlex content)
python scripts/process_pdf_corpus_with_grobid.py   # PDF → TEI XML (GROBID)
python scripts/parse_tei_corpus.py
python scripts/parse_jats_corpus.py
python scripts/assemble_parsed_corpus.py
python scripts/assemble_final_corpus.py    # data/processed/final_corpus.jsonl
python scripts/chunk_corpus.py             # data/processed/document_chunks.jsonl
python scripts/build_retrieval_indexes.py  # data/processed/indexes/{dense.faiss,retrieval.sqlite3}
```

Publication dates for time-bound questions are read from
`final_corpus.jsonl` at load time, so changing them does not require
re-embedding.

## Ask a question

```bash
python scripts/run_agentic_graph.py "Why does CRAG strip web search results before generation?"

# Or open the graph in LangGraph Studio
langgraph dev
```

Baseline-only and component-only entry points exist for debugging:
`scripts/ask_baseline_rag.py`, `scripts/run_corrective_retrieval.py`,
`scripts/run_self_rag.py`, `scripts/run_web_fallback.py`,
`scripts/search_hybrid.py`.

## Evaluate

```bash
# Smoke test: two pipelines, three questions, no judge model
python scripts/run_evaluation.py --pipelines baseline,agentic --limit 3 --skip-ragas

# Full ablation, deterministic metrics only (resumable)
python scripts/run_evaluation.py --skip-ragas

# Add RAGAS scores to the saved answers (only missing metrics are scored)
python scripts/score_evaluation_with_ragas.py
```

Outputs are `data/benchmark/results/evaluation_records.jsonl` (one record per
pipeline-question pair) and `evaluation_report.json` (per-pipeline means).
Pass `--no-resume` to discard the saved records and start over.

Retrieval gold labels were produced by pooled judging
(`scripts/judge_retrieval_candidates.py`); see `docs/evaluation.md` for the
procedure, metric definitions and how to extend the benchmark.

### Results

Last full run on the original 50 questions, `gpt-4.1-mini` for generation and
judging. Retrieval metrics in that run covered only 4 hand-labelled questions;
the benchmark now carries gold for all 64 static retrieval questions plus 21
harder questions, so the next run will report them over the whole set.

| Pipeline | Success | Faithfulness | Answer relevancy | Context recall | Factual correctness | Latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.00 | 0.79 | 0.93 | 0.75 | 0.48 | 4.4 |
| crag | 0.90 | 0.83 | 0.93 | 0.77 | 0.45 | 10.9 |
| crag_web | 0.94 | 0.80 | 0.93 | 0.80 | 0.42 | 11.0 |
| agentic | 0.98 | 0.83 | 0.95 | 0.85 | 0.46 | 15.2 |

Reading: the agentic graph has the best grounding and context recall, at about
3.5x the latency of the single-pass baseline. The original 50 questions were
easy enough that the graph took the happy path on 45 of them, which is why the
21 hard questions were added.

## Tests and lint

```bash
pytest -q
ruff check src tests scripts
```

Tests use fakes for the LLM, retriever and web search and need no API keys or
index files. The same two commands run in CI on every pull request.

## Project layout

```
configs/            corpus.yaml (ingestion, retrieval, graph), evaluation.yaml
data/benchmark/     agentic_rag_evaluation.jsonl, retrieval_judgments.jsonl, results/
docs/               evaluation.md
scripts/            command-line entry points for every stage
src/agentic_rag/    library code
tests/              pytest suite
```
