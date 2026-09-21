import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_openalex_api_key():
    api_key = os.getenv("OPENALEX_API_KEY")
    if not api_key:
        raise ValueError("OPENALEX_API_KEY is not set")
    return api_key


def resolve_config_path(config_path=None):
    if config_path is None:
        return PROJECT_ROOT / "configs" / "corpus.yaml"

    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_corpus_config(config_path=None):
    path = resolve_config_path(config_path)

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_ollama_grading_config(config_path=None):
    config = load_corpus_config(config_path)

    if "ollama_grading" not in config:
        raise KeyError(
            "Missing 'ollama_grading' section in "
            + str(resolve_config_path(config_path))
        )

    return config["ollama_grading"]
