import json
import time
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ValidationError

from agentic_rag.processing.ollama_grader import (
    append_jsonl,
    create_ollama_client,
    get_abstract,
    paper_key,
    read_jsonl,
    write_json,
    write_jsonl,
)


SYSTEM_PROMPT = """
You are a conservative first-stage research-paper screener.

The final corpus covers textual information retrieval and
retrieval-augmented generation.

Your only task is to identify papers that are CLEARLY unrelated.

Clear rejects include papers whose central topic is obviously:
- computer vision or image classification;
- medical or biological research;
- remote sensing;
- materials or molecular simulation;
- wireless networking;
- robotics or point-cloud processing;
- general machine learning without textual retrieval;
- another field with no textual search, retrieval, RAG, or retrieval-based QA.

Never reject a paper when its title or abstract preview suggests:
- retrieval-augmented generation or RAG;
- text, passage, document, dense, sparse, hybrid, or neural retrieval;
- document or passage ranking and reranking;
- cross-encoder retrieval;
- query rewriting, expansion, routing, or decomposition;
- open-domain or multi-hop question answering;
- grounding, citation verification, or retrieval-based hallucination control;
- uncertainty about whether retrieval is central.

This is a high-recall stage. False rejection is much worse than keeping an
irrelevant paper. If evidence is incomplete or uncertain, keep the paper.

Return only zero-based indexes of papers that are clearly unrelated.
Do not return explanations.
""".strip()


class BatchScreenResult(BaseModel):
    clear_reject_indexes: list[int] = Field(default_factory=list)


def build_screening_items(papers, preview_characters):
    items = []

    for index, paper in enumerate(papers):
        title = paper.get("title") or ""
        abstract = get_abstract(paper)

        if len(abstract) > preview_characters:
            abstract = abstract[:preview_characters]

        item = {
            "index": index,
            "title": title.strip(),
            "abstract_preview": abstract.strip(),
        }

        items.append(item)

    return items


def build_screening_prompt(items):
    serialized_items = json.dumps(
        items,
        ensure_ascii=False,
        indent=2,
    )

    return (
        "Screen the following papers. Return only indexes that are "
        "clearly unrelated to textual retrieval or RAG.\n\n"
        + serialized_items
    )


def normalize_reject_indexes(indexes, batch_size):
    normalized = []

    for index in indexes:
        if isinstance(index, bool):
            continue

        if not isinstance(index, int):
            continue

        if index < 0 or index >= batch_size:
            continue

        if index not in normalized:
            normalized.append(index)

    return normalized


def screen_batch(papers, client, screening_config):
    items = build_screening_items(
        papers,
        screening_config["preview_characters"],
    )

    user_prompt = build_screening_prompt(items)
    last_error = None

    for attempt_number in range(
        1,
        screening_config["max_retries"] + 1,
    ):
        try:
            response = client.chat(
                model=screening_config["model"],
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                format=BatchScreenResult.model_json_schema(),
                options={
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": screening_config["context_window"],
                    "num_predict": screening_config["num_predict"],
                },
                think=False,
                stream=False,
                keep_alive=screening_config["keep_alive"],
            )

            result = BatchScreenResult.model_validate_json(
                response.message.content
            )

            return normalize_reject_indexes(
                result.clear_reject_indexes,
                len(papers),
            )

        except (
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

        except Exception as error:
            last_error = error

        if attempt_number < screening_config["max_retries"]:
            wait_seconds = (
                screening_config["retry_delay_seconds"]
                * attempt_number
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        "Ollama batch screening failed after "
        + str(screening_config["max_retries"])
        + " attempts: "
        + str(last_error)
    )


def add_screening_decision(
    paper,
    decision,
    model_name,
    fallback_keep=False,
):
    screened_paper = dict(paper)

    screened_paper["ollama_screening"] = {
        "model": model_name,
        "decision": decision,
        "fallback_keep": fallback_keep,
        "screened_at": datetime.now(timezone.utc).isoformat(),
    }

    return screened_paper


def load_paper_keys(path):
    keys = set()
    records = read_jsonl(path)

    for paper in records:
        keys.add(paper_key(paper))

    return keys


def build_screening_outputs(screening_config):
    candidates = read_jsonl(
        screening_config["input_path"]
    )

    graded_keys = load_paper_keys(
        screening_config["existing_graded_path"]
    )

    decision_records = read_jsonl(
        screening_config["decisions_output_path"]
    )

    candidate_keys = set()

    for paper in candidates:
        candidate_keys.add(paper_key(paper))

    decisions_by_key = {}

    for paper in decision_records:
        key = paper_key(paper)

        if key not in candidate_keys:
            continue

        if key in graded_keys:
            continue

        decisions_by_key[key] = paper

    shortlist = []
    rejected = []
    fallback_kept = 0
    previously_graded = 0

    for paper in candidates:
        key = paper_key(paper)

        if key in graded_keys:
            previously_graded += 1
            continue

        screened_paper = decisions_by_key.get(key)

        if screened_paper is None:
            continue

        screening = screened_paper.get(
            "ollama_screening",
            {},
        )

        decision = screening.get("decision")

        if decision == "clear_reject":
            rejected.append(screened_paper)
        else:
            shortlist.append(screened_paper)

            if screening.get("fallback_keep"):
                fallback_kept += 1

    write_jsonl(
        shortlist,
        screening_config["shortlist_output_path"],
    )

    write_jsonl(
        rejected,
        screening_config["rejected_output_path"],
    )

    eligible_candidates = (
        len(candidates) - previously_graded
    )

    screened_count = len(shortlist) + len(rejected)

    remaining_unscreened = max(
        eligible_candidates - screened_count,
        0,
    )

    status = "partial"

    if remaining_unscreened == 0:
        status = "complete"

    report = {
        "status": status,
        "model": screening_config["model"],
        "total_ambiguous_candidates": len(candidates),
        "previously_graded": previously_graded,
        "eligible_for_screening": eligible_candidates,
        "screened": screened_count,
        "shortlisted_for_full_grading": len(shortlist),
        "clear_rejects": len(rejected),
        "fallback_kept": fallback_kept,
        "remaining_unscreened": remaining_unscreened,
        "screening_keep_rate": (
            len(shortlist) / screened_count
            if screened_count
            else 0.0
        ),
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    write_json(
        report,
        screening_config["report_path"],
    )

    return report


def screen_ambiguous_corpus(
    screening_config,
    limit=None,
):
    candidates = read_jsonl(
        screening_config["input_path"]
    )

    graded_keys = load_paper_keys(
        screening_config["existing_graded_path"]
    )

    processed_keys = load_paper_keys(
        screening_config["decisions_output_path"]
    )

    pending = []

    for paper in candidates:
        key = paper_key(paper)

        if key in graded_keys:
            continue

        if key in processed_keys:
            continue

        pending.append(paper)

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        return build_screening_outputs(
            screening_config
        )

    client = create_ollama_client(
        screening_config
    )

    batch_size = screening_config["batch_size"]
    newly_screened = 0

    for batch_start in range(
        0,
        len(pending),
        batch_size,
    ):
        batch = pending[
            batch_start:batch_start + batch_size
        ]

        fallback_keep = False

        try:
            reject_indexes = screen_batch(
                batch,
                client,
                screening_config,
            )

        except RuntimeError as error:
            print(
                "Screening batch failed; keeping every "
                "paper in this batch:",
                error,
            )

            reject_indexes = []
            fallback_keep = True

        reject_index_set = set(reject_indexes)

        for index, paper in enumerate(batch):
            decision = "keep"

            if index in reject_index_set:
                decision = "clear_reject"

            screened_paper = add_screening_decision(
                paper,
                decision,
                screening_config["model"],
                fallback_keep=fallback_keep,
            )

            append_jsonl(
                screened_paper,
                screening_config[
                    "decisions_output_path"
                ],
            )

            newly_screened += 1

        print(
            "Newly screened:",
            newly_screened,
            "| Selected this run:",
            len(pending),
        )

    return build_screening_outputs(
        screening_config
    )