import json
from collections import Counter
from pathlib import Path


RELEVANCE_TERMS = {
    "retrieval augmented generation": 8,
    "retrieval-augmented generation": 8,
    "corrective rag": 8,
    "self-rag": 8,
    "self rag": 8,
    "agentic retrieval": 7,
    "large language model": 4,
    "language model agent": 5,
    "dense retrieval": 4,
    "sparse retrieval": 4,
    "hybrid retrieval": 4,
    "neural information retrieval": 4,
    "information retrieval": 3,
    "document retrieval": 3,
    "passage retrieval": 3,
    "text retrieval": 4,
    "text ranking": 4,
    "document ranking": 4,
    "passage ranking": 4,
    "listwise reranking": 4,
    "retrieval reranking": 4,
    "re-ranking": 2,
    "reranking": 2,
    "cross encoder": 3,
    "query rewriting": 3,
    "query expansion": 2,
    "question answering": 3,
    "open-domain question answering": 5,
    "open domain question answering": 5,
    "multi-hop question answering": 5,
    "multi hop question answering": 5,
    "semantic search": 3,
    "retrieval grounding": 4,
    "hallucination detection": 3,
    "citation verification": 3,
}


OFF_DOMAIN_TITLE_TERMS = [
    "image retrieval",
    "video retrieval",
    "visual retrieval",
    "text-to-visual",
    "multimodal retrieval",
    "multi-modal retrieval",
    "cross-modal retrieval",
    "instance retrieval",
    "visual information retrieval",
    "visual analytics",
    "visual recognition",
    "vision-language",
    "computer vision",
    "multimedia",
    "person reidentification",
    "person re-identification",
    "remote sensing",
    "satellite",
    "sentinel",
    "forest",
    "grass",
    "crop",
    "chlorophyll",
    "vegetation",
    "x-ray",
    "medical imaging",
    "medical image",
    "liver lesion",
    "phase retrieval",
    "orbital imaging",
    "exoplanet",
    "protein",
    "genome",
    "genomic",
    "music retrieval",
    "audio signal",
    "recommender system",
    "transliteration",
    "sentence simplification",
    "bilingual lexicon",
    "bridge maintenance",
    "time series",
    "subspace clustering",
    "long-read assembly",
    "action recognition",
    "currency strategies",
    "tag-cloud",
]

LLM_TERMS = [
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "large language model",
    "language model agent",
    "corrective rag",
    "self-rag",
    "self rag",
    "agentic retrieval",
]

TITLE_DOMAIN_TERMS = [
    "retrieval augmented",
    "retrieval-augmented",
    "rag-augmented",
    "rag system",
    "retrieval augmented language model",
    "retrieval-augmented langu  age model",
    "retrieval augmented text generation",
    "retrieval-augmented text generation",
    "neural ranking",
    "cross-lingual retrieval",
    "zero-shot retrieval",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "corrective rag",
    "self-rag",
    "self rag",
    "agentic retrieval",
    "large language model",
    "language model agent",
    "dense retrieval",
    "sparse retrieval",
    "hybrid retrieval",
    "neural information retrieval",
    "information retrieval",
    "document retrieval",
    "passage retrieval",
    "text retrieval",
    "text ranking",
    "document ranking",
    "passage ranking",
    "listwise reranking",
    "retrieval reranking",
    "re-ranking",
    "reranking",
    "cross encoder",
    "query rewriting",
    "query expansion",
    "question answering",
    "semantic search",
]


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


def build_metadata_text(paper):
    values = []

    primary_topic = paper.get("primary_topic")

    if primary_topic:
        topic_name = primary_topic.get("display_name")

        if topic_name:
            values.append(topic_name)

    topics = paper.get("topics", [])

    for topic in topics:
        topic_name = topic.get("display_name")

        if topic_name:
            values.append(topic_name)

    keywords = paper.get("keywords", [])

    for keyword in keywords:
        keyword_name = keyword.get("display_name")

        if keyword_name:
            values.append(keyword_name)

    matched_queries = paper.get(
        "matched_queries",
        [],
    )

    for query in matched_queries:
        values.append(query)

    return " ".join(values).lower()


def contains_any(text, terms):
    for term in terms:
        if term in text:
            return True

    return False


def score_relevance(paper):
    title = paper.get("title") or ""
    title = title.lower()

    abstract = reconstruct_abstract(paper.get("abstract_inverted_index"))
    abstract = abstract.lower()

    metadata_text = build_metadata_text(paper)

    score = 0

    title_terms = []
    abstract_terms = []
    metadata_terms = []

    for term, weight in RELEVANCE_TERMS.items():
        if term in title:
            score += weight * 2
            title_terms.append(term)

        elif term in abstract:
            score += weight
            abstract_terms.append(term)

        elif term in metadata_text:
            metadata_weight = max(
                1,
                weight // 2,
            )

            score += metadata_weight
            metadata_terms.append(term)

    query_match_count = paper.get(
        "query_match_count",
        0,
    )

    score += min(
        query_match_count,
        3,
    )

    penalty_terms = []

    has_llm_title_signal = contains_any(
        title,
        LLM_TERMS,
    )
    has_domain_title_signal = contains_any(
        title,
        TITLE_DOMAIN_TERMS,
    )

    if not has_llm_title_signal:
        for term in OFF_DOMAIN_TITLE_TERMS:
            if term in title:
                penalty_terms.append(term)

        if penalty_terms:
            score -= 12

    if not has_domain_title_signal:
        if score >= 8:
            score = 7

    if score < 0:
        score = 0

    matched_terms = []

    for term in title_terms:
        matched_terms.append(term)

    for term in abstract_terms:
        matched_terms.append(term)

    for term in metadata_terms:
        matched_terms.append(term)

    return {
        "score": score,
        "matched_terms": matched_terms,
        "title_terms": title_terms,
        "abstract_terms": abstract_terms,
        "metadata_terms": metadata_terms,
        "penalty_terms": penalty_terms,
    }


def find_quality_failures(
    paper,
    selection_config,
):
    failures = []

    if paper.get("is_retracted"):
        failures.append("retracted")

    if not paper.get("title"):
        failures.append("missing_title")

    if selection_config["require_abstract"]:
        abstract = paper.get("abstract_inverted_index")

        if not abstract:
            failures.append("missing_abstract")

    if selection_config["require_downloadable_content"]:
        has_content = paper.get("has_content") or {}

        has_pdf = has_content.get(
            "pdf",
            False,
        )

        has_xml = has_content.get(
            "grobid_xml",
            False,
        )

        if not has_pdf and not has_xml:
            failures.append("missing_downloadable_content")

    return failures


def evaluate_candidate(
    paper,
    selection_config,
):
    relevance = score_relevance(paper)

    quality_failures = find_quality_failures(
        paper,
        selection_config,
    )

    retain_score = selection_config["retain_score"]

    ambiguous_score = selection_config["ambiguous_score"]

    decision = "rejected"

    if not quality_failures:
        if relevance["score"] >= retain_score:
            decision = "retained"

        elif relevance["score"] >= ambiguous_score:
            decision = "ambiguous"

    evaluated_paper = dict(paper)

    evaluated_paper["selection"] = {
        "decision": decision,
        "relevance_score": relevance["score"],
        "title_terms": relevance["title_terms"],
        "abstract_terms": relevance["abstract_terms"],
        "metadata_terms": relevance["metadata_terms"],
        "penalty_terms": relevance["penalty_terms"],
        "quality_failures": quality_failures,
        "matched_terms": relevance["matched_terms"],
    }

    return evaluated_paper


def candidate_sort_key(paper):
    selection = paper["selection"]

    relevance_score = selection["relevance_score"]

    cited_by_count = paper.get("cited_by_count") or 0

    publication_year = paper.get("publication_year") or 0

    has_content = paper.get("has_content") or {}

    has_xml = int(
        has_content.get(
            "grobid_xml",
            False,
        )
    )

    return (
        relevance_score,
        has_xml,
        cited_by_count,
        publication_year,
    )


def score_sort_key(item):
    return int(item[0])


def write_jsonl(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            json.dump(
                record,
                output_file,
                ensure_ascii=False,
            )

            output_file.write("\n")


def filter_candidates(
    config,
    input_path,
    output_paths,
):
    selection_config = config["selection"]

    retained = []
    ambiguous = []
    rejected = []

    rejection_reasons = Counter()
    score_distribution = Counter()

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            paper = json.loads(line)

            evaluated_paper = evaluate_candidate(
                paper,
                selection_config,
            )

            selection = evaluated_paper["selection"]

            decision = selection["decision"]

            score = selection["relevance_score"]

            score_distribution[str(score)] += 1

            if decision == "retained":
                retained.append(evaluated_paper)

            elif decision == "ambiguous":
                ambiguous.append(evaluated_paper)

            else:
                rejected.append(evaluated_paper)

                failures = selection["quality_failures"]

                if failures:
                    for failure in failures:
                        rejection_reasons[failure] += 1
                else:
                    rejection_reasons["low_relevance"] += 1

    retained.sort(
        key=candidate_sort_key,
        reverse=True,
    )

    ambiguous.sort(
        key=candidate_sort_key,
        reverse=True,
    )

    write_jsonl(
        retained,
        output_paths["retained"],
    )

    write_jsonl(
        ambiguous,
        output_paths["ambiguous"],
    )

    write_jsonl(
        rejected,
        output_paths["rejected"],
    )

    total_candidates = len(retained) + len(ambiguous) + len(rejected)

    target_papers = config["corpus"]["target_papers"]

    sorted_scores = sorted(
        score_distribution.items(),
        key=score_sort_key,
    )

    report = {
        "total_candidates": total_candidates,
        "retained": len(retained),
        "ambiguous": len(ambiguous),
        "rejected": len(rejected),
        "target_papers": target_papers,
        "difference_from_target": (len(retained) - target_papers),
        "rejection_reasons": dict(rejection_reasons),
        "score_distribution": dict(sorted_scores),
    }

    report_path = Path(output_paths["report"])

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    return report
