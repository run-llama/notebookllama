import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.notebookllama.models import Notebook
from src.notebookllama.utils import create_llama_extract_client
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    conn = create_llama_extract_client()
    agent = conn.create_agent(name="q_and_a_agent", data_schema=Notebook)
    _id = agent.id
    with open(".env", "a") as f:
        f.write(f'\nEXTRACT_AGENT_ID="{_id}"')
    return 0


if __name__ == "__main__":
    main()
