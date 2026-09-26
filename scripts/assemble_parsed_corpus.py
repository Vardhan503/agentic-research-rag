import json
from pathlib import Path
from typing import Any

from agentic_rag.config import load_corpus_config
from agentic_rag.processing.parsed_corpus import (
    assemble_parsed_corpus,
)


def get_unified_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Get the unified parsed-corpus configuration."""

    extraction_config = project_config.get(
        "text_extraction"
    )

    if extraction_config is None:
        raise KeyError(
            "Missing text_extraction configuration."
        )

    unified_config = extraction_config.get("unified")

    if unified_config is None:
        raise KeyError(
            "Missing text_extraction.unified configuration."
        )

    return unified_config


def main() -> None:
    """Assemble JATS and TEI papers into one corpus."""

    project_config = load_corpus_config()
    config = get_unified_config(project_config)

    report = assemble_parsed_corpus(
        corpus_path=Path(config["corpus_path"]),
        parsed_jats_path=Path(
            config["parsed_jats_path"]
        ),
        parsed_tei_path=Path(
            config["parsed_tei_path"]
        ),
        output_path=Path(config["output_path"]),
        report_path=Path(config["report_path"]),
    )

    print()
    print("=" * 80)
    print("PARSED CORPUS REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print()


if __name__ == "__main__":
    main()