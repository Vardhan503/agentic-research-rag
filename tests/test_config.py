from agentic_rag.config import load_corpus_config


def test_corpus_target():
    config = load_corpus_config()

    target_papers = config["corpus"]["target_papers"]

    assert target_papers == 10000


def test_domain_queries_exist():
    config = load_corpus_config()

    queries = config["domain"]["search_queries"]

    assert len(queries) > 0