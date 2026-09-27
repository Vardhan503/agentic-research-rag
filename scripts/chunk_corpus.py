import json
from pathlib import Path
from typing import Any

from agentic_rag.config import load_corpus_config
from agentic_rag.processing.chunk_pipeline import (
    chunk_corpus_file,
)


def get_chunking_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Read and validate the chunking configuration."""

    chunking_config = project_config.get("chunking")

    if chunking_config is None:
        raise KeyError(
            "The chunking section is missing from "
            "configs/corpus.yaml."
        )

    required_settings = (
        "input_path",
        "output_path",
        "report_path",
        "max_tokens",
        "overlap_tokens",
    )

    for setting in required_settings:
        if setting not in chunking_config:
            raise KeyError(
                f"Missing chunking setting: {setting}"
            )

    return chunking_config


def main() -> None:
    """Chunk every paper in the parsed corpus."""

    project_config = load_corpus_config()
    config = get_chunking_config(project_config)

    print()
    print("=" * 80)
    print("DOCUMENT CHUNKING")
    print("=" * 80)
    print()
    print(f"Input: {config['input_path']}")
    print(f"Maximum tokens: {config['max_tokens']}")
    print(
        f"Overlap tokens: "
        f"{config['overlap_tokens']}"
    )
    print()

    report = chunk_corpus_file(
        input_path=Path(config["input_path"]),
        output_path=Path(config["output_path"]),
        report_path=Path(config["report_path"]),
        max_tokens=int(config["max_tokens"]),
        overlap_tokens=int(
            config["overlap_tokens"]
        ),
        include_abstract=bool(
            config.get("include_abstract", True)
        ),
        small_chunk_threshold=int(
            config.get(
                "small_chunk_threshold",
                40,
            )
        ),
        progress_interval=int(
            config.get("progress_interval", 100)
        ),
    )

    print()
    print("=" * 80)
    print("CHUNKING REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print()


if __name__ == "__main__":
    main()