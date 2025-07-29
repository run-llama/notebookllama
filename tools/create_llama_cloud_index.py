import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from dotenv import load_dotenv
from cli.embedding_app import EmbeddingSetupApp
from src.notebookllama.utils import create_llamacloud_client

from llama_cloud import (
    PipelineTransformConfig_Advanced,
    AdvancedModeTransformConfigChunkingConfig_Sentence,
    AdvancedModeTransformConfigSegmentationConfig_Page,
    PipelineCreate,
)


def main():
    """
    Create a new Llama Cloud index with the given embedding configuration.
    """
    load_dotenv()
    client = create_llamacloud_client()

    app = EmbeddingSetupApp()
    embedding_config = app.run()

    if embedding_config:
        segm_config = AdvancedModeTransformConfigSegmentationConfig_Page(mode="page")
        chunk_config = AdvancedModeTransformConfigChunkingConfig_Sentence(
            chunk_size=1024,
            chunk_overlap=200,
            separator="<whitespace>",
            paragraph_separator="\n\n\n",
            mode="sentence",
        )

        transform_config = PipelineTransformConfig_Advanced(
            segmentation_config=segm_config,
            chunking_config=chunk_config,
            mode="advanced",
        )

        pipeline_request = PipelineCreate(
            name="notebooklm_pipeline",
            embedding_config=embedding_config,
            transform_config=transform_config,
        )

        pipeline = asyncio.run(
            client.pipelines.upsert_pipeline(request=pipeline_request)
        )

        with open(".env", "a") as f:
            f.write(f'\nLLAMACLOUD_PIPELINE_ID="{pipeline.id}"')

        return 0
    else:
        print("No embedding configuration provided")
        return 1


if __name__ == "__main__":
    main()
