import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ollama import Client
from pydantic import BaseModel, Field, ValidationError


SYSTEM_PROMPT = """
You are a strict research-paper screener building a high-precision corpus about
textual information retrieval and retrieval-augmented generation (RAG).

Decide whether the paper's CENTRAL RESEARCH CONTRIBUTION belongs in this corpus.
Use only the supplied title and abstract. Do not infer relevance from where the
paper was discovered, generic keywords, citations, or possible future uses.

A paper is relevant when its main contribution directly studies at least one:
- retrieval-augmented generation, corrective RAG, Self-RAG, or agentic retrieval;
- text, document, passage, dense, sparse, hybrid, or neural retrieval;
- search ranking, passage ranking, document ranking, or retrieval reranking;
- cross-encoder reranking for text retrieval;
- query rewriting, expansion, routing, or decomposition for retrieval;
- open-domain or multi-hop question answering that explicitly retrieves text;
- grounding, citation verification, or hallucination control using retrieved text.

A paper is irrelevant when retrieval is only incidental, background motivation,
a possible application, a cited related work item, or a minor evaluation detail.
Also mark irrelevant:
- general LLM/model technical reports;
- general prompting, AI, machine learning, or deep-learning surveys;
- generic healthcare, education, taxonomy, knowledge-graph, or continual-learning
  papers without a primary textual-retrieval contribution;
- image, video, audio, remote-sensing, phase, biological, or recommendation
  retrieval unless the paper's primary task is textual retrieval for RAG or QA;
- closed-book question answering without a retrieval/search component.

Use ambiguous only when the title or abstract provides direct evidence that
textual retrieval may be a primary contribution, but the available abstract is
insufficient to decide safely. Do not use ambiguous merely because a paper is
about AI, NLP, embeddings, transformers, LLMs, or question answering.

Evidence rules:
1. The title and the stated objective/contributions in the abstract are strongest.
2. A relevant reason must name the exact retrieval/RAG contribution.
3. A keyword mention by itself is not enough.
4. When the central contribution is clearly outside scope, choose irrelevant.

Score calibration:
- relevant: relevance_score from 0.80 to 1.00
- ambiguous: relevance_score from 0.40 to 0.79
- irrelevant: relevance_score from 0.00 to 0.39

Return only the requested structured result. Keep the reason under 35 words.
""".strip()


class PaperGrade(BaseModel):
    label: Literal["relevant", "ambiguous", "irrelevant"]
    relevance_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=3, max_length=300)


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""

    largest_position = -1

    for positions in inverted_index.values():
        for position in positions:
            if position > largest_position:
                largest_position = position

    words = [""] * (largest_position + 1)

    for word, positions in inverted_index.items():
        for position in positions:
            words[position] = word

    return " ".join(words).strip()


def get_abstract(paper):
    abstract = paper.get("abstract")

    if abstract:
        return str(abstract).strip()

    inverted_index = paper.get("abstract_inverted_index")
    return reconstruct_abstract(inverted_index)


def build_paper_payload(paper, max_abstract_characters):
    title = paper.get("title") or ""
    abstract = get_abstract(paper)

    if len(abstract) > max_abstract_characters:
        abstract = abstract[:max_abstract_characters]

    # Intentionally exclude deterministic scores, matched terms, OpenAlex
    # topics/keywords, and discovery queries. Those fields caused label leakage.
    return {
        "title": title.strip(),
        "abstract": abstract.strip(),
    }


def build_user_prompt(paper_payload):
    title = paper_payload["title"]
    abstract = paper_payload["abstract"]

    if not abstract:
        abstract = "[Abstract unavailable]"

    return (
        "Classify this paper for the textual retrieval and RAG corpus.\n\n"
        "Title:\n"
        + title
        + "\n\nAbstract:\n"
        + abstract
    )


def validate_grade_consistency(grade):
    score = grade.relevance_score

    if grade.label == "relevant" and score < 0.80:
        raise ValueError("Relevant label must have a score of at least 0.80")

    if grade.label == "ambiguous":
        if score < 0.40 or score >= 0.80:
            raise ValueError("Ambiguous score must be between 0.40 and 0.79")

    if grade.label == "irrelevant" and score >= 0.40:
        raise ValueError("Irrelevant label must have a score below 0.40")


def paper_key(paper):
    work_id = paper.get("id") or paper.get("paper_id")

    if work_id:
        return str(work_id)

    doi = paper.get("doi")

    if doi:
        return str(doi).lower().strip()

    title = paper.get("title") or ""
    normalized_title = " ".join(title.lower().split())
    return "title:" + normalized_title


def grade_paper(paper, client, grading_config):
    payload = build_paper_payload(
        paper,
        grading_config["max_abstract_characters"],
    )

    user_prompt = build_user_prompt(payload)
    last_error = None

    for attempt_number in range(1, grading_config["max_retries"] + 1):
        try:
            response = client.chat(
                model=grading_config["model"],
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
                format=PaperGrade.model_json_schema(),
                options={
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": grading_config["context_window"],
                    "num_predict": grading_config["num_predict"],
                },
                think=False,
                stream=False,
                keep_alive=grading_config["keep_alive"],
            )

            grade = PaperGrade.model_validate_json(
                response.message.content
            )

            validate_grade_consistency(grade)
            return grade

        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            last_error = error

        except Exception as error:
            last_error = error

        if attempt_number < grading_config["max_retries"]:
            wait_seconds = (
                grading_config["retry_delay_seconds"] * attempt_number
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        "Ollama grading failed after "
        + str(grading_config["max_retries"])
        + " attempts: "
        + str(last_error)
    )


def route_grade(grade, grading_config):
    if grade.label == "relevant":
        if grade.relevance_score >= grading_config["accepted_min_score"]:
            return "accepted"

    if grade.label == "irrelevant":
        if grade.relevance_score <= grading_config["rejected_max_score"]:
            return "rejected"

    return "needs_review"


def add_grade_to_paper(paper, grade, route, model_name):
    graded_paper = dict(paper)

    graded_paper["ollama_grade"] = {
        "model": model_name,
        "label": grade.label,
        "relevance_score": grade.relevance_score,
        "reason": grade.reason,
        "route": route,
        "graded_at": datetime.now(timezone.utc).isoformat(),
    }

    return graded_paper


def read_jsonl(path):
    records = []
    input_path = Path(path)

    if not input_path.exists():
        return records

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            records.append(json.loads(stripped_line))

    return records


def append_jsonl(record, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as output_file:
        json.dump(record, output_file, ensure_ascii=False)
        output_file.write("\n")
        output_file.flush()


def write_jsonl(records, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for record in records:
            json.dump(record, output_file, ensure_ascii=False)
            output_file.write("\n")


def write_json(data, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def create_ollama_client(grading_config):
    return Client(
        host=grading_config["host"],
        timeout=grading_config["timeout_seconds"],
    )


def load_processed_keys(graded_output_path):
    processed_keys = set()
    graded_records = read_jsonl(graded_output_path)

    for paper in graded_records:
        processed_keys.add(paper_key(paper))

    return processed_keys


def write_checkpoint(grading_config, processed_count, total_selected):
    checkpoint = {
        "model": grading_config["model"],
        "processed": processed_count,
        "total_selected": total_selected,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    write_json(checkpoint, grading_config["checkpoint_path"])


def split_graded_outputs(grading_config):
    graded_records = read_jsonl(grading_config["graded_output_path"])

    accepted = []
    needs_review = []
    rejected = []

    for paper in graded_records:
        route = paper.get("ollama_grade", {}).get("route")

        if route == "accepted":
            accepted.append(paper)
        elif route == "rejected":
            rejected.append(paper)
        else:
            needs_review.append(paper)

    write_jsonl(accepted, grading_config["accepted_output_path"])
    write_jsonl(needs_review, grading_config["review_output_path"])
    write_jsonl(rejected, grading_config["rejected_output_path"])

    return accepted, needs_review, rejected


def read_selection_counts(grading_config):
    report_path = Path(grading_config["selection_report_path"])

    if not report_path.exists():
        return 0, 0

    with open(report_path, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    retained = int(report.get("retained", 0))
    target = int(report.get("target_papers", 0))
    return retained, target


def create_grading_report(grading_config):
    accepted, needs_review, rejected = split_graded_outputs(grading_config)
    deterministic_retained, target_papers = read_selection_counts(
        grading_config
    )

    potential_final_corpus = deterministic_retained + len(accepted)
    remaining_gap = max(target_papers - potential_final_corpus, 0)

    report = {
        "model": grading_config["model"],
        "total_graded": len(accepted) + len(needs_review) + len(rejected),
        "accepted": len(accepted),
        "needs_review": len(needs_review),
        "rejected": len(rejected),
        "deterministic_retained": deterministic_retained,
        "potential_final_corpus": potential_final_corpus,
        "target_papers": target_papers,
        "remaining_gap": remaining_gap,
    }

    write_json(report, grading_config["report_path"])
    return report


def grade_ambiguous_corpus(grading_config, limit=None):
    candidates = read_jsonl(grading_config["input_path"])

    if limit is not None:
        candidates = candidates[:limit]

    selected_keys = set()

    for paper in candidates:
        selected_keys.add(paper_key(paper))

    processed_keys = load_processed_keys(
        grading_config["graded_output_path"]
    )

    completed_selected = len(
        selected_keys.intersection(
            processed_keys
        )
    )

    client = create_ollama_client(grading_config)
    newly_graded = 0

    for paper in candidates:
        key = paper_key(paper)

        if key in processed_keys:
            continue

        grade = grade_paper(paper, client, grading_config)
        route = route_grade(grade, grading_config)

        graded_paper = add_grade_to_paper(
            paper,
            grade,
            route,
            grading_config["model"],
        )

        append_jsonl(
            graded_paper,
            grading_config["graded_output_path"],
        )

        processed_keys.add(key)
        newly_graded += 1
        completed_selected += 1

        if newly_graded % grading_config["batch_size"] == 0:
            print(
                "Newly graded:",
                newly_graded,
                "| Shortlist completed:",
                completed_selected,
                "| Shortlist selected:",
                len(candidates),
            )

            write_checkpoint(
                grading_config,
                completed_selected,
                len(candidates),
            )

    write_checkpoint(
        grading_config,
        completed_selected,
        len(candidates),
    )

    return create_grading_report(grading_config)

