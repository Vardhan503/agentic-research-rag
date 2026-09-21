import json

from agentic_rag.config import (
    load_ollama_screening_config,
)
from agentic_rag.processing.batch_screener import (
    screen_batch,
)
from agentic_rag.processing.ollama_grader import (
    create_ollama_client,
)


POSITIVE_CONTROLS = [
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "We combine a neural retriever with a sequence generator that conditions on retrieved Wikipedia passages.",
    },
    {
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "abstract": "We train dense question and passage encoders for retrieving relevant text passages for open-domain question answering.",
    },
    {
        "title": "ColBERT: Efficient and Effective Passage Search",
        "abstract": "We introduce contextualized late interaction for efficient neural passage retrieval and ranking.",
    },
    {
        "title": "SPLADE: Sparse Lexical and Expansion Model for Information Retrieval",
        "abstract": "We learn sparse document and query representations for first-stage text retrieval.",
    },
    {
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique",
        "abstract": "The language model retrieves passages on demand and critiques whether generated responses are supported by retrieved evidence.",
    },
    {
        "title": "Corrective Retrieval Augmented Generation",
        "abstract": "We evaluate retrieved documents and trigger corrective retrieval when initial evidence is unreliable.",
    },
    {
        "title": "Query2doc: Query Expansion with Large Language Models",
        "abstract": "We generate pseudo-documents to expand search queries and improve sparse and dense document retrieval.",
    },
    {
        "title": "Passage Re-ranking with BERT",
        "abstract": "We use a cross-encoder to rerank candidate passages returned by a first-stage text retrieval system.",
    },
    {
        "title": "Anserini: Enabling the Use of Lucene for Information Retrieval Research",
        "abstract": "We present a reproducible toolkit for document indexing, retrieval, and ranking experiments.",
    },
    {
        "title": "Multi-Hop Retrieval for Open-Domain Question Answering",
        "abstract": "We retrieve multiple supporting text passages across reasoning steps to answer open-domain questions.",
    },
]


NEGATIVE_CONTROLS = [
    {
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "abstract": "We train a convolutional neural network to classify high-resolution images.",
    },
    {
        "title": "CheXpert: A Large Chest Radiograph Dataset",
        "abstract": "We introduce a dataset for medical image interpretation and chest radiograph classification.",
    },
    {
        "title": "LAMMPS: A Flexible Simulation Tool for Materials Modeling",
        "abstract": "We describe molecular dynamics simulation software for particle-based materials modeling.",
    },
    {
        "title": "Dynamic Graph CNN for Learning on Point Clouds",
        "abstract": "We introduce a graph neural network architecture for point-cloud classification and segmentation.",
    },
    {
        "title": "VINS-Mono: A Monocular Visual-Inertial State Estimator",
        "abstract": "We present a visual-inertial odometry system for autonomous robotics.",
    },
    {
        "title": "Remote Sensing of Forest Chlorophyll",
        "abstract": "Satellite imagery is used to estimate vegetation and forest chlorophyll levels.",
    },
    {
        "title": "Diagnosis and Management of Dementia with Lewy Bodies",
        "abstract": "We provide clinical recommendations for diagnosis and treatment of dementia.",
    },
    {
        "title": "Re-epithelialization in an Ex Vivo Human Skin Model",
        "abstract": "We study immune-cell behavior and biological wound-healing processes.",
    },
    {
        "title": "EEGNet for Brain-Computer Interfaces",
        "abstract": "We develop a convolutional network for classification of electroencephalography signals.",
    },
    {
        "title": "A Review of Landslide Susceptibility Models",
        "abstract": "We review statistical methods for geographic landslide susceptibility assessment.",
    },
]


def main():
    config = load_ollama_screening_config()
    client = create_ollama_client(config)

    controls = []

    for paper in POSITIVE_CONTROLS:
        controls.append(paper)

    for paper in NEGATIVE_CONTROLS:
        controls.append(paper)

    rejected_indexes = screen_batch(
        controls,
        client,
        config,
    )

    rejected_index_set = set(rejected_indexes)

    positive_rejected = []
    negative_rejected = []

    for index, paper in enumerate(
        POSITIVE_CONTROLS
    ):
        if index in rejected_index_set:
            positive_rejected.append(
                paper["title"]
            )

    negative_start = len(
        POSITIVE_CONTROLS
    )

    for offset, paper in enumerate(
        NEGATIVE_CONTROLS
    ):
        combined_index = (
            negative_start + offset
        )

        if combined_index in rejected_index_set:
            negative_rejected.append(
                paper["title"]
            )

    positive_kept = (
        len(POSITIVE_CONTROLS)
        - len(positive_rejected)
    )

    report = {
        "positive_controls": len(
            POSITIVE_CONTROLS
        ),
        "positive_controls_kept": positive_kept,
        "positive_controls_rejected": len(
            positive_rejected
        ),
        "positive_recall": (
            positive_kept
            / len(POSITIVE_CONTROLS)
        ),
        "negative_controls": len(
            NEGATIVE_CONTROLS
        ),
        "negative_controls_rejected": len(
            negative_rejected
        ),
        "negative_rejection_rate": (
            len(negative_rejected)
            / len(NEGATIVE_CONTROLS)
        ),
        "false_rejected_positive_titles": (
            positive_rejected
        ),
        "rejected_negative_titles": (
            negative_rejected
        ),
    }

    print(json.dumps(report, indent=2))

    if positive_rejected:
        raise SystemExit(
            "Validation failed: at least one "
            "positive control was rejected."
        )

    print(
        "\nValidation passed: every known "
        "retrieval paper was kept."
    )


if __name__ == "__main__":
    main()