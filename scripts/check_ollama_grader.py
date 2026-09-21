from typing import Literal

from ollama import chat
from pydantic import BaseModel
from pydantic import Field


MODEL_NAME = "qwen3:4b-instruct"


SYSTEM_PROMPT = """
You classify research papers for a corpus about
retrieval-augmented generation and text retrieval.

Relevant subjects include:
- retrieval-augmented generation
- Corrective RAG and Self-RAG
- text, document, and passage retrieval
- dense, sparse, and hybrid text retrieval
- retrieval reranking
- query rewriting
- question answering
- LLM agents
- grounding
- hallucination detection
- citation verification

Exclude papers primarily about:
- image or video retrieval
- remote sensing
- atmospheric retrieval
- medical imaging
- bioinformatics or protein retrieval
- phase retrieval
- recommendation systems without textual retrieval

Use:
- relevant: clearly useful for our RAG corpus
- ambiguous: possibly useful but insufficient evidence
- irrelevant: outside our target domain

Return a short reason based on the title and abstract.
"""


class PaperGrade(BaseModel):
    label: Literal[
        "relevant",
        "ambiguous",
        "irrelevant",
    ]

    confidence: float = Field(
        ge=0,
        le=1,
    )

    reason: str


def grade_paper(title, abstract):
    user_prompt = f"Title: {title}\n\nAbstract: {abstract}"

    response = chat(
        model=MODEL_NAME,
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
        },
    )

    grade = PaperGrade.model_validate_json(response.message.content)

    return grade


def main():
    relevant_grade = grade_paper(
        title=("Corrective Retrieval-Augmented Generation for Large Language Models"),
        abstract=(
            "We evaluate retrieved passages, "
            "rewrite weak queries, and use web "
            "search when document retrieval fails."
        ),
    )

    irrelevant_grade = grade_paper(
        title=("Hybrid Retrieval of Forest Chlorophyll from Satellite Images"),
        abstract=("We estimate vegetation properties using remote-sensing satellite data."),
    )

    print()
    print("Relevant example:")
    print(relevant_grade.model_dump_json(indent=2))

    print()
    print("Irrelevant example:")
    print(irrelevant_grade.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
