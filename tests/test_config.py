from agentic_rag.config import load_corpus_config


def test_corpus_target():
    config = load_corpus_config()

    target_papers = config["corpus"]["target_papers"]

    assert target_papers == 10000


def test_domain_queries_exist():
    config = load_corpus_config()

    queries = config["domain"]["search_queries"]

    assert len(queries) > 0


def test_ollama_grading_config_paths():
    from agentic_rag.config import load_ollama_grading_config

    grading_config = load_ollama_grading_config()

    assert grading_config["model"] == "qwen3:14b"
    assert grading_config["input_path"] == (
        "data/interim/ambiguous_screened_shortlist.jsonl"
    )
    assert grading_config["graded_output_path"].endswith("ambiguous_graded.jsonl")
