import os
from typing import Optional, Dict, Any
from llama_cloud.client import AsyncLlamaCloud
from llama_cloud_services import LlamaExtract, LlamaParse
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# LlamaCloud regional endpoints
LLAMACLOUD_REGIONS = {
    "default": "https://api.cloud.llamaindex.ai",  # North America (default)
    "eu": "https://api.cloud.eu.llamaindex.ai",  # Europe
}


class LlamaCloudConfigError(Exception):
    """Raised when LlamaCloud configuration is invalid."""

    pass


def get_llamacloud_base_url() -> Optional[str]:
    """
    Get the appropriate LlamaCloud base URL based on region configuration.

    Defaults to North America region and returns the North America endpoint URL
    when no region is specified.

    Returns:
        str: The base URL for LlamaCloud API

    Raises:
        LlamaCloudConfigError: If an invalid region is specified
    """
    # Direct base URL override takes precedence
    base_url = os.getenv("LLAMACLOUD_BASE_URL")
    if base_url:
        return base_url

    region = os.getenv("LLAMACLOUD_REGION", "default").lower().strip()

    if region not in LLAMACLOUD_REGIONS:
        valid_regions = ", ".join(LLAMACLOUD_REGIONS.keys())
        raise LlamaCloudConfigError(
            f"Invalid LLAMACLOUD_REGION '{region}'. Supported regions: {valid_regions}"
        )

    return LLAMACLOUD_REGIONS[region]


def get_llamacloud_config() -> Dict[str, Any]:
    """
    Get LlamaCloud configuration including base URL.

    Returns:
        dict: Configuration dictionary with token and optional base_url

    Raises:
        LlamaCloudConfigError: If API key is missing or region is invalid
    """
    token = os.getenv("LLAMACLOUD_API_KEY")
    if not token:
        raise LlamaCloudConfigError(
            "LLAMACLOUD_API_KEY environment variable is required"
        )

    config = {"token": token}

    base_url = get_llamacloud_base_url()
    if base_url:
        config["base_url"] = base_url

    return config


def create_llamacloud_client() -> AsyncLlamaCloud:
    """
    Create a configured AsyncLlamaCloud client with regional support.

    Returns:
        AsyncLlamaCloud: Configured client instance

    Raises:
        LlamaCloudConfigError: If API key is missing or region is invalid
    """
    config = get_llamacloud_config()
    return AsyncLlamaCloud(**config)


def create_llama_extract_client() -> LlamaExtract:
    """
    Create a configured LlamaExtract client with regional support.

    Returns:
        LlamaExtract: Configured client instance

    Raises:
        LlamaCloudConfigError: If API key is missing or region is invalid
    """
    api_key = os.getenv("LLAMACLOUD_API_KEY")
    if not api_key:
        raise LlamaCloudConfigError(
            "LLAMACLOUD_API_KEY environment variable is required"
        )

    base_url = get_llamacloud_base_url()
    return LlamaExtract(api_key=api_key, base_url=base_url)


def create_llama_parse_client(result_type: str = "markdown") -> LlamaParse:
    """
    Create a configured LlamaParse client with regional support.

    Args:
        result_type: The result type for parsing (default: "markdown")

    Returns:
        LlamaParse: Configured client instance

    Raises:
        LlamaCloudConfigError: If API key is missing or region is invalid
    """
    api_key = os.getenv("LLAMACLOUD_API_KEY")
    if not api_key:
        raise LlamaCloudConfigError(
            "LLAMACLOUD_API_KEY environment variable is required"
        )

    base_url = get_llamacloud_base_url()
    return LlamaParse(api_key=api_key, result_type=result_type, base_url=base_url)


def create_llamacloud_index(api_key: str, pipeline_id: str) -> LlamaCloudIndex:
    """
    Create a configured LlamaCloudIndex with regional support.

    Args:
        api_key: The API key for authentication
        pipeline_id: The pipeline ID to use

    Returns:
        LlamaCloudIndex: Configured index instance

    Raises:
        LlamaCloudConfigError: If API key or pipeline_id is missing, or region is invalid
    """
    if not api_key:
        raise LlamaCloudConfigError("API key is required")

    if not pipeline_id:
        raise LlamaCloudConfigError("Pipeline ID is required")

    base_url = get_llamacloud_base_url()
    return LlamaCloudIndex(api_key=api_key, pipeline_id=pipeline_id, base_url=base_url)
