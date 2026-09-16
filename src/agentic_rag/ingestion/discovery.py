import json
import time
from collections import Counter
from pathlib import Path

from agentic_rag.ingestion.openalex_client import OpenAlexClient


def build_filters(config):
    corpus_config = config["corpus"]

    language = corpus_config["language"]

    minimum_year = corpus_config["publication_year"]["min"]
    maximum_year = corpus_config["publication_year"]["max"]

    work_types = corpus_config["work_types"]

    type_filter = "|".join(work_types)

    filters = []

    filters.append(f"language:{language}")

    filters.append(
        f"from_publication_date:{minimum_year}-01-01"
    )

    filters.append(
        f"to_publication_date:{maximum_year}-12-31"
    )

    filters.append(
        f"type:{type_filter}"
    )

    if corpus_config["require_open_access"]:
        filters.append("open_access.is_oa:true")

    if corpus_config["require_full_text"]:
        filters.append("has_fulltext:true")

    return ",".join(filters)


def discover_raw_candidates(
    config,
    raw_output_path,
    candidate_target,
    max_results_per_query,
):
    client = OpenAlexClient()

    discovery_config = config["discovery"]
    domain_config = config["domain"]

    queries = domain_config["search_queries"]

    per_page = discovery_config["per_page"]

    delay = discovery_config["request_delay_seconds"]

    filters = build_filters(config)

    raw_output_path = Path(raw_output_path)

    raw_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_raw_records = 0
    total_api_cost = 0.0

    unique_ids_seen = set()

    print()
    print("=" * 80)
    print("OPENALEX DOMAIN DISCOVERY")
    print("=" * 80)

    print()
    print("Filters:")
    print(filters)

    print()
    print("Candidate target:")
    print(candidate_target)

    with open(
        raw_output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        stop_discovery = False

        for query in queries:
            print()
            print("-" * 80)
            print("Query:")
            print(query)

            cursor = "*"

            query_result_count = 0
            query_page = 0

            while cursor:
                if query_result_count >= max_results_per_query:
                    break

                if len(unique_ids_seen) >= candidate_target:
                    stop_discovery = True
                    break

                query_page += 1

                response = client.search_works_page(
                    query=query,
                    filters=filters,
                    cursor=cursor,
                    per_page=per_page,
                )

                metadata = response.get(
                    "meta",
                    {},
                )

                page_cost = metadata.get(
                    "cost_usd",
                    0,
                )

                if page_cost:
                    total_api_cost += float(page_cost)

                papers = response.get(
                    "results",
                    [],
                )

                if not papers:
                    break

                for paper in papers:
                    if query_result_count >= max_results_per_query:
                        break

                    paper_id = paper.get("id")

                    raw_record = {
                        "discovery_query": query,
                        "query_rank": query_result_count + 1,
                        "work": paper,
                    }

                    json.dump(
                        raw_record,
                        output_file,
                        ensure_ascii=False,
                    )

                    output_file.write("\n")

                    total_raw_records += 1
                    query_result_count += 1

                    if paper_id:
                        unique_ids_seen.add(paper_id)

                    if len(unique_ids_seen) >= candidate_target:
                        stop_discovery = True
                        break

                cursor = metadata.get(
                    "next_cursor"
                )

                print(
                    f"Page {query_page} | "
                    f"query records: {query_result_count} | "
                    f"unique papers: {len(unique_ids_seen)}"
                )

                if stop_discovery:
                    break

                time.sleep(delay)

            print()
            print(
                "Completed query with "
                f"{query_result_count} records."
            )

            if stop_discovery:
                break

    result = {
        "raw_records": total_raw_records,
        "unique_ids_seen": len(unique_ids_seen),
        "api_cost_usd": round(total_api_cost, 6),
    }

    return result


def deduplicate_candidates(
    raw_input_path,
    unique_output_path,
):
    raw_input_path = Path(raw_input_path)
    unique_output_path = Path(unique_output_path)

    unique_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = {}

    doi_to_paper_id = {}

    raw_records = 0

    with open(
        raw_input_path,
        "r",
        encoding="utf-8",
    ) as input_file:

        for line in input_file:
            raw_records += 1

            record = json.loads(line)

            query = record["discovery_query"]
            paper = record["work"]

            paper_id = paper.get("id")
            doi = paper.get("doi")

            existing_id = None

            if paper_id in candidates:
                existing_id = paper_id

            elif doi:
                if doi in doi_to_paper_id:
                    existing_id = doi_to_paper_id[doi]

            if existing_id:
                existing_paper = candidates[existing_id]

                matched_queries = existing_paper[
                    "matched_queries"
                ]

                if query not in matched_queries:
                    matched_queries.append(query)

                continue

            candidate = dict(paper)

            candidate["matched_queries"] = [
                query
            ]

            candidate["query_match_count"] = 1

            candidates[paper_id] = candidate

            if doi:
                doi_to_paper_id[doi] = paper_id

    with open(
        unique_output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        for paper_id in candidates:
            candidate = candidates[paper_id]

            candidate["query_match_count"] = len(
                candidate["matched_queries"]
            )

            json.dump(
                candidate,
                output_file,
                ensure_ascii=False,
            )

            output_file.write("\n")

    duplicates_removed = (
        raw_records - len(candidates)
    )

    result = {
        "raw_records": raw_records,
        "unique_papers": len(candidates),
        "duplicates_removed": duplicates_removed,
    }

    return result


def create_discovery_report(
    unique_input_path,
    report_output_path,
):
    unique_input_path = Path(
        unique_input_path
    )

    report_output_path = Path(
        report_output_path
    )

    report_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    year_counts = Counter()
    query_counts = Counter()
    topic_counts = Counter()

    total_papers = 0

    papers_with_doi = 0
    papers_with_abstract = 0

    papers_with_pdf = 0
    papers_with_xml = 0

    retracted_papers = 0

    with open(
        unique_input_path,
        "r",
        encoding="utf-8",
    ) as input_file:

        for line in input_file:
            paper = json.loads(line)

            total_papers += 1

            year = paper.get(
                "publication_year"
            )

            if year:
                year_counts[str(year)] += 1

            doi = paper.get("doi")

            if doi:
                papers_with_doi += 1

            abstract = paper.get(
                "abstract_inverted_index"
            )

            if abstract:
                papers_with_abstract += 1

            has_content = paper.get(
                "has_content"
            )

            if has_content:
                if has_content.get("pdf"):
                    papers_with_pdf += 1

                if has_content.get(
                    "grobid_xml"
                ):
                    papers_with_xml += 1

            if paper.get("is_retracted"):
                retracted_papers += 1

            matched_queries = paper.get(
                "matched_queries",
                [],
            )

            for query in matched_queries:
                query_counts[query] += 1

            topics = paper.get(
                "topics",
                [],
            )

            for topic in topics:
                topic_name = topic.get(
                    "display_name"
                )

                if topic_name:
                    topic_counts[topic_name] += 1

    most_common_topics = []

    for topic_name, count in topic_counts.most_common(30):
        topic_record = {
            "topic": topic_name,
            "count": count,
        }

        most_common_topics.append(
            topic_record
        )

    report = {
        "total_unique_papers": total_papers,
        "papers_with_doi": papers_with_doi,
        "papers_with_abstract": papers_with_abstract,
        "papers_with_pdf": papers_with_pdf,
        "papers_with_grobid_xml": papers_with_xml,
        "retracted_papers": retracted_papers,
        "year_distribution": dict(
            sorted(year_counts.items())
        ),
        "query_distribution": dict(
            query_counts
        ),
        "top_topics": most_common_topics,
    }

    with open(
        report_output_path,
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