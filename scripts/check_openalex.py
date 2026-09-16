from agentic_rag.ingestion.openalex_client import OpenAlexClient


def print_paper(paper):
    print()
    print("=" * 80)

    print("Title:")
    print(paper.get("title"))

    print()
    print("OpenAlex ID:")
    print(paper.get("id"))

    print()
    print("DOI:")
    print(paper.get("doi"))

    print()
    print("Publication year:")
    print(paper.get("publication_year"))

    print()
    print("Type:")
    print(paper.get("type"))

    print()
    print("Citation count:")
    print(paper.get("cited_by_count"))

    print()
    print("Has full text:")
    print(paper.get("has_fulltext"))

    print()
    print("Content availability:")
    print(paper.get("has_content"))

    primary_topic = paper.get("primary_topic")

    if primary_topic:
        print()
        print("Primary topic:")
        print(primary_topic.get("display_name"))

    print()
    print("Referenced works:")
    references = paper.get("referenced_works")

    if references:
        print("Number of references:", len(references))
    else:
        print("Number of references: 0")


def main():
    client = OpenAlexClient()

    query = "retrieval augmented generation"

    print()
    print("Searching OpenAlex...")
    print("Query:", query)

    response = client.search_works(
        query=query,
        per_page=5,
    )

    metadata = response.get("meta", {})

    print()
    print("Total matching works:")
    print(f"metadata: {metadata}")
    print(metadata.get("count"))

    papers = response.get("results", [])

    print()
    print("Returned papers:")
    print(len(papers))

    for paper in papers:
        print_paper(paper)


if __name__ == "__main__":
    main()