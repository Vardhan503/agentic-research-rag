import json
from typing import Any

from agentic_rag.config import load_corpus_config
from agentic_rag.processing.corpus_audit import audit_local_corpus


def get_audit_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Get and validate the corpus audit configuration."""

    audit_config = project_config.get("corpus_audit")

    if audit_config is None:
        raise KeyError(
            "The corpus_audit section is missing from "
            "configs/corpus.yaml."
        )

    required_settings = (
        "corpus_path",
        "manifest_path",
        "pdf_directory",
        "xml_directory",
        "inventory_output",
        "report_output",
    )

    for setting in required_settings:
        if setting not in audit_config:
            raise KeyError(
                f"Missing corpus_audit setting: {setting}"
            )

    return audit_config


def print_audit_plan(
    audit_config: dict[str, Any],
) -> None:
    """Show which corpus and directories will be audited."""

    print()
    print("=" * 80)
    print("LOCAL CORPUS AUDIT")
    print("=" * 80)
    print()
    print(f"Final corpus: {audit_config['corpus_path']}")
    print(f"PDF directory: {audit_config['pdf_directory']}")
    print(f"XML directory: {audit_config['xml_directory']}")
    print()
    print("This audit does not modify or delete downloaded files.")
    print()


def print_report(report: dict[str, Any]) -> None:
    """Print the completed audit report as readable JSON."""

    print()
    print("=" * 80)
    print("LOCAL CORPUS AUDIT REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2))
    print()


def main() -> None:
    """Load configuration, run the audit, and display results."""

    project_config = load_corpus_config()
    audit_config = get_audit_config(project_config)

    print_audit_plan(audit_config)

    report = audit_local_corpus(audit_config)

    print_report(report)


if __name__ == "__main__":
    main()