from agentic_rag.processing.corpus_selection import (
    evaluate_candidate,
    reconstruct_abstract,
)


def make_paper(title, abstract_words):
    inverted_index = {}

    position = 0

    for word in abstract_words.split():
        positions = inverted_index.get(
            word,
            [],
        )

        positions.append(position)
        inverted_index[word] = positions

        position += 1

    return {
        "id": "https://openalex.org/W1",
        "title": title,
        "abstract_inverted_index": (inverted_index),
        "matched_queries": ["dense retrieval"],
        "query_match_count": 1,
        "has_content": {
            "pdf": True,
            "grobid_xml": True,
        },
        "is_retracted": False,
    }


def make_config():
    return {
        "retain_score": 8,
        "ambiguous_score": 4,
        "require_abstract": True,
        "require_downloadable_content": True,
    }


def test_reconstruct_abstract_orders_words():
    inverted_index = {
        "retrieval": [1],
        "agentic": [0],
        "works": [2],
    }

    abstract = reconstruct_abstract(inverted_index)

    assert abstract == ("agentic retrieval works")


def test_rag_paper_is_retained():
    paper = make_paper(
        ("Corrective Retrieval-Augmented Generation"),
        "large language model grounding",
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "retained"


def test_remote_sensing_paper_is_rejected():
    paper = make_paper(
        "Dense retrieval for remote sensing",
        "image retrieval for satellite images",
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "rejected"

    assert "remote sensing" in result["selection"]["penalty_terms"]


def test_retracted_paper_is_rejected():
    paper = make_paper(
        "Retrieval augmented generation",
        "large language model retrieval",
    )

    paper["is_retracted"] = True

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "rejected"

    assert "retracted" in result["selection"]["quality_failures"]


def test_valid_dense_retrieval_is_not_penalized():
    paper = make_paper(
        "A Thorough Examination on Zero-shot Dense Retrieval",
        ("The evaluation compares text retrieval with image retrieval and video retrieval"),
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "retained"

    assert result["selection"]["penalty_terms"] == []


def test_exoplanet_retrieval_is_rejected():
    paper = make_paper(
        ("Hybrid Retrieval of Exoplanetary Emission Spectra"),
        "A retrieval method for planetary science",
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "rejected"

    assert "exoplanet" in result["selection"]["penalty_terms"]


def test_forest_retrieval_is_rejected():
    paper = make_paper(
        ("Estimating Forest Leaf Area Index with Hybrid Retrieval Algorithms"),
        "Machine learning and regression methods",
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "rejected"

    assert "forest" in result["selection"]["penalty_terms"]


def test_abstract_only_match_is_not_retained():
    paper = make_paper(
        "Language-agnostic BERT Sentence Embedding",
        ("The model supports text retrieval and information retrieval"),
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "ambiguous"


def test_generic_reranking_is_not_auto_retained():
    paper = make_paper(
        ("Reranking Individuals: Fair Classification Within Groups"),
        "A fairness classification method",
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] != "retained"


def test_rag_augmented_title_is_retained():
    paper = make_paper(
        ("A RAG-Augmented LLM for Domain Question Answering"),
        ("A large language model with retrieval augmented generation"),
    )

    result = evaluate_candidate(
        paper,
        make_config(),
    )

    assert result["selection"]["decision"] == "retained"
