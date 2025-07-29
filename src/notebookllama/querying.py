import os
import sys
from dotenv import load_dotenv

from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.llms.openai import OpenAIResponses
from typing import Union, cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from notebookllama.utils import create_llamacloud_index

load_dotenv()

if (
    os.getenv("LLAMACLOUD_API_KEY", None)
    and os.getenv("LLAMACLOUD_PIPELINE_ID", None)
    and os.getenv("OPENAI_API_KEY", None)
):
    LLM = OpenAIResponses(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
    PIPELINE_ID = os.getenv("LLAMACLOUD_PIPELINE_ID")
    API_KEY = os.getenv("LLAMACLOUD_API_KEY")

    if API_KEY is None or PIPELINE_ID is None:
        raise ValueError("LLAMACLOUD_API_KEY and LLAMACLOUD_PIPELINE_ID must be set")

    index = create_llamacloud_index(api_key=API_KEY, pipeline_id=PIPELINE_ID)
    RETR = index.as_retriever()
    QE = CitationQueryEngine(
        retriever=RETR,
        llm=LLM,
        citation_chunk_size=256,
        citation_chunk_overlap=50,
    )


async def query_index(question: str) -> Union[str, None]:
    response = await QE.aquery(question)
    response = cast(Response, response)
    sources = []
    if not response.response:
        return None
    if response.source_nodes is not None:
        sources = [node.text for node in response.source_nodes]
    return (
        "## Answer\n\n"
        + response.response
        + "\n\n## Sources\n\n- "
        + "\n- ".join(sources)
    )
