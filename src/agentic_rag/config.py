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

def load_corpus_config():
    config_path = PROJECT_ROOT / "configs" / "corpus.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config