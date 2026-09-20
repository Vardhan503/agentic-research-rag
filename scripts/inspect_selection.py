import json


RETAINED_PATH = "data/processed/corpus_test_selected.jsonl"

AMBIGUOUS_PATH = "data/interim/corpus_test_ambiguous.jsonl"

REJECTED_PATH = "data/interim/corpus_test_rejected.jsonl"


def load_records(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            record = json.loads(line)
            records.append(record)

    return records


def get_score(paper):
    return paper["selection"]["relevance_score"]


def print_paper(paper):
    selection = paper["selection"]

    print()
    print("Title:")
    print(
        paper.get(
            "title",
            "Missing title",
        )
    )

    print("Score:")
    print(selection["relevance_score"])

    print("Title terms:")
    print(
        selection.get(
            "title_terms",
            [],
        )
    )

    print("Abstract terms:")
    print(
        selection.get(
            "abstract_terms",
            [],
        )
    )

    print("Metadata terms:")
    print(
        selection.get(
            "metadata_terms",
            [],
        )
    )

    print("Penalty terms:")
    print(
        selection.get(
            "penalty_terms",
            [],
        )
    )

    print("Discovery queries:")
    print(
        paper.get(
            "matched_queries",
            [],
        )
    )

    print("-" * 80)


def print_group(
    heading,
    records,
    reverse,
    limit=10,
):
    records.sort(
        key=get_score,
        reverse=reverse,
    )

    print()
    print("=" * 80)
    print(heading)
    print("=" * 80)

    count = 0

    for paper in records:
        print_paper(paper)

        count += 1

        if count >= limit:
            break


def main():
    retained = load_records(RETAINED_PATH)

    ambiguous = load_records(AMBIGUOUS_PATH)

    rejected = load_records(REJECTED_PATH)

    print_group(
        heading="WEAKEST RETAINED PAPERS",
        records=retained,
        reverse=False,
    )

    print_group(
        heading="HIGHEST-SCORING AMBIGUOUS PAPERS",
        records=ambiguous,
        reverse=True,
    )

    print_group(
        heading="HIGHEST-SCORING REJECTED PAPERS",
        records=rejected,
        reverse=True,
    )


if __name__ == "__main__":
    main()
