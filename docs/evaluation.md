# Agentic RAG evaluation

This evaluation compares four systems on the same versioned 71-question test
set (50 original questions plus 21 hard ones: deep-detail, multi-hop,
wrong-premise, time-bound, out-of-corpus and unanswerable):

1. `baseline`: retrieve once and generate once.
2. `crag`: grade evidence and rewrite weak retrieval before generation.
3. `crag_web`: add web fallback when the local corpus remains insufficient.
4. `agentic`: run the complete graph with routing, correction, grounding checks,
   usefulness checks, and bounded retries.

## Metrics

The deterministic metrics do not call an evaluator model:

- `precision_at_k`: relevant items divided by retrieved items in the first k.
- `recall_at_k`: known relevant items recovered in the first k.
- `reciprocal_rank`: reciprocal rank of the first relevant item.
- `ndcg_at_k`: position-discounted ranked relevance.
- `citation_validity`: cited chunks that were actually retrieved.
- `citation_recall`: known relevant sources that were cited.
- `keyword_coverage`: required `must_contain` phrases found in the answer,
  minus any `must_not_contain` phrases (used so a stale 2024 paper cannot
  score fully on a "last seven days" question).
- `route_accuracy`: correctness of retrieval and web-routing decisions.
- `success`: a non-empty, error-free answer with an accepted terminal status.
  Questions marked `expects_abstention` count as successful only when the
  system returns `insufficient_evidence` instead of inventing an answer.
  Questions marked `accepts_abstention` also count `insufficient_evidence` as
  success, because a dated answer and an honest abstention are both correct.
- `latency_seconds`: end-to-end wall-clock time per question.

RAGAS adds semantic evaluator-model metrics:

- `faithfulness`: answer claims supported by retrieved context.
- `answer_relevancy`: how directly the answer addresses the question.
- `context_precision`: how much retrieved context is useful rather than noise.
- `context_recall`: how much reference information the context covers.
- `factual_correctness`: claim-level agreement with the reference answer.

Reference-dependent metrics are skipped for dynamic web questions because their
correct answer changes over time.

## Install evaluation dependencies

```bash
pip install -r requirements-evaluation.txt
```

The `langchain-community==0.4.1` pin is intentional. RAGAS 0.4.3 currently
imports a compatibility module that was removed in version 0.4.2.

## Run the tests

```bash
pytest
```

Tests use fakes and do not call Ollama, Tavily, LangSmith, or the embedding
model.

## Recommended staged evaluation

First run two questions without RAGAS. This validates retrieval, generation,
checkpointing, and report creation quickly:

```bash
python scripts/run_evaluation.py \
  --pipelines baseline,agentic \
  --limit 2 \
  --skip-ragas
```

If the smoke test succeeds, run all four pipelines while still skipping RAGAS:

```bash
python scripts/run_evaluation.py --skip-ragas
```

This creates:

- `data/benchmark/results/evaluation_records.jsonl`: one detailed record for
  each pipeline-question pair.
- `data/benchmark/results/evaluation_report.json`: aggregate comparison.

Both steps resume automatically. Do not use `--no-resume` unless you
intentionally want to replace the current experiment.

Next test RAGAS on five already-saved answers:

```bash
python scripts/score_evaluation_with_ragas.py --limit 5
```

Then score all remaining saved answers:

```bash
python scripts/score_evaluation_with_ragas.py
```

This second command does not rerun any RAG pipeline. It updates and checkpoints
the saved records after each local evaluator call.

For the slowest, publication-quality answer-relevancy score, change
`answer_relevancy_strictness` from `1` to `3` in `configs/evaluation.yaml`, use a
new results path, and rerun the experiment.

## Retrieval gold labels

Every static retrieval question carries `expected_paper_ids`, so
`recall_at_k`, `reciprocal_rank` and `ndcg_at_k` are computed over the whole
benchmark. Labels come from two sources:

1. Hand-picked anchor papers for the 21 hard questions (the paper the
   reference answer was written from).
2. Pooled judging for the original 50 questions: the hybrid retriever's top-20
   chunks are judged chunk by chunk by `gpt-4.1-mini`, which sees the question
   *and the reference answer* and must say whether the chunk supports that
   answer. A paper is relevant when any of its chunks is. The judgments and the
   judge's one-line reasons are committed in
   `data/benchmark/retrieval_judgments.jsonl` for review.

Pooled labels only cover papers the retriever can already find, so `recall_at_k`
is an upper bound and the ranking metrics (`reciprocal_rank`, `ndcg_at_k`)
are the more trustworthy comparison between pipelines.

Regenerate or extend the labels after changing the benchmark:

```bash
# Judge only questions that have no saved judgment, then merge into the benchmark.
python scripts/judge_retrieval_candidates.py --reuse --apply

# Re-judge everything from scratch (about 50 judge calls).
python scripts/judge_retrieval_candidates.py --apply
```

`--apply` unions the pooled relevant papers with any existing
`expected_paper_ids` and appends a provenance note to the question's `notes`.
Questions with `requires_retrieval: false` or `reference_mode: dynamic` are
skipped.

## Optional LangSmith tracing and dataset upload

Set the environment variables before running the graph:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-key"
export LANGSMITH_PROJECT="agentic-research-rag-evaluation"
```

Upload or safely update the versioned benchmark dataset:

```bash
python scripts/upload_evaluation_dataset.py
```

The uploader uses deterministic example IDs, so rerunning it updates the same
examples rather than intentionally creating a second benchmark.

## Reading the final comparison

Do not select a system from one average alone. Prefer the simplest pipeline
that meets the required faithfulness and retrieval quality. Compare its
faithfulness, factual correctness, recall, citation validity, failure rate, and
latency together. The agentic graph is justified only when its quality gain is
large enough to offset its additional model calls and latency.
