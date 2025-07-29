import pytest
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock

from typing import Callable
from pydantic import ValidationError
from src.notebookllama.processing import (
    process_file,
    md_table_to_pd_dataframe,
    rename_and_remove_current_images,
    rename_and_remove_past_images,
    MarkdownTextAnalyzer,
)
from src.notebookllama.mindmap import get_mind_map
from src.notebookllama.models import Notebook
from src.notebookllama.utils import (
    get_llamacloud_base_url,
    get_llamacloud_config,
    create_llamacloud_client,
    create_llama_extract_client,
    create_llama_parse_client,
    create_llamacloud_index,
    LlamaCloudConfigError,
    LLAMACLOUD_REGIONS,
)

load_dotenv()

skip_condition = not (
    os.getenv("LLAMACLOUD_API_KEY", None)
    and os.getenv("EXTRACT_AGENT_ID", None)
    and os.getenv("LLAMACLOUD_PIPELINE_ID", None)
    and os.getenv("OPENAI_API_KEY", None)
)


@pytest.fixture()
def input_file() -> str:
    return "data/test/brain_for_kids.pdf"


@pytest.fixture()
def markdown_file() -> str:
    return "data/test/md_sample.md"


@pytest.fixture()
def images_dir() -> str:
    return "data/test/images/"


@pytest.fixture()
def dataframe_from_tables() -> pd.DataFrame:
    project_data = {
        "Project Name": [
            "User Dashboard",
            "API Integration",
            "Mobile App",
            "Database Migration",
            "Security Audit",
        ],
        "Status": [
            "In Progress",
            "Completed",
            "Planning",
            "In Progress",
            "Not Started",
        ],
        "Completion %": ["75%", "100%", "25%", "60%", "0%"],
        "Assigned Developer": [
            "Alice Johnson",
            "Bob Smith",
            "Carol Davis",
            "David Wilson",
            "Eve Brown",
        ],
        "Due Date": [
            "2025-07-15",
            "2025-06-30",
            "2025-08-20",
            "2025-07-10",
            "2025-08-01",
        ],
    }

    df = pd.DataFrame(project_data)
    return df


@pytest.fixture()
def file_exists_fn() -> Callable[[str], bool]:
    def file_exists(file_path: str) -> bool:
        return Path(file_path).exists()

    return file_exists


@pytest.fixture()
def is_not_empty_fn() -> Callable[[str], bool]:
    def is_not_empty(file_path: str) -> bool:
        return Path(file_path).stat().st_size > 0

    return is_not_empty


@pytest.fixture
def notebook_to_process() -> Notebook:
    return Notebook(
        summary="""The Human Brain:
        The human brain is a complex organ responsible for thought, memory, emotion, and coordination. It contains about 86 billion neurons and operates through electrical and chemical signals. Divided into major parts like the cerebrum, cerebellum, and brainstem, it controls everything from basic survival functions to advanced reasoning. Despite its size, it consumes around 20% of the body’s energy. Neuroscience continues to explore its mysteries, including consciousness and neuroplasticity—its ability to adapt and reorganize.""",
        questions=[
            "How many neurons are in the human brain?",
            "What are the main parts of the human brain?",
            "What percentage of the body's energy does the brain use?",
            "What is neuroplasticity?",
            "What functions is the human brain responsible for?",
        ],
        answers=[
            "About 86 billion neurons.",
            "The cerebrum, cerebellum, and brainstem.",
            "Around 20%.",
            "The brain's ability to adapt and reorganize itself.",
            "Thought, memory, emotion, and coordination.",
        ],
        highlights=[
            "The human brain has about 86 billion neurons.",
            "It controls thought, memory, emotion, and coordination.",
            "Major brain parts include the cerebrum, cerebellum, and brainstem.",
            "The brain uses approximately 20% of the body's energy.",
            "Neuroplasticity allows the brain to adapt and reorganize.",
        ],
    )


@pytest.mark.skipif(
    condition=skip_condition,
    reason="You do not have the necessary env variables to run this test.",
)
@pytest.mark.asyncio
async def test_mind_map_creation(
    notebook_to_process: Notebook,
    file_exists_fn: Callable[[str], bool],
    is_not_empty_fn: Callable[[str], bool],
):
    test_mindmap = await get_mind_map(
        summary=notebook_to_process.summary, highlights=notebook_to_process.highlights
    )
    assert test_mindmap is not None
    assert file_exists_fn(test_mindmap)
    assert is_not_empty_fn(test_mindmap)
    os.remove(test_mindmap)


@pytest.mark.skipif(
    condition=skip_condition,
    reason="You do not have the necessary env variables to run this test.",
)
@pytest.mark.asyncio
async def test_file_processing(input_file: str) -> None:
    notebook, text = await process_file(filename=input_file)
    print(notebook)
    assert notebook is not None
    assert isinstance(text, str)
    try:
        notebook_model = Notebook.model_validate_json(json_data=notebook)
    except ValidationError:
        notebook_model = None
    assert isinstance(notebook_model, Notebook)


def test_table_to_dataframe(
    markdown_file: str, dataframe_from_tables: pd.DataFrame
) -> None:
    with open(markdown_file, "r") as f:
        text = f.read()
    analyzer = MarkdownTextAnalyzer(text)
    md_tables = analyzer.identify_tables()["Table"]
    assert len(md_tables) == 2
    for md_table in md_tables:
        df = md_table_to_pd_dataframe(md_table)
        assert df is not None
        assert df.equals(dataframe_from_tables)


def test_images_renaming(images_dir: str):
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    imgs = rename_and_remove_current_images(images)
    assert all("_current" in img for img in imgs)
    assert all(os.path.exists(img) for img in imgs)
    renamed = rename_and_remove_past_images(images_dir)
    assert all("_at_" in img for img in renamed)
    assert all("_current" not in img for img in renamed)
    assert all(os.path.exists(img) for img in renamed)
    for image in renamed:
        with open(image, "rb") as rb:
            bts = rb.read()
        with open(images_dir + "image.png", "wb") as wb:
            wb.write(bts)
        os.remove(image)


# =============================================================================
# Regional LlamaCloud Utilities Tests
# =============================================================================


class TestLlamaCloudRegionalUtils:
    """Test suite for regional LlamaCloud utility functions."""

    def test_llamacloud_regions_constant(self):
        """Test that LLAMACLOUD_REGIONS contains expected regions."""
        assert "default" in LLAMACLOUD_REGIONS
        assert "eu" in LLAMACLOUD_REGIONS
        assert LLAMACLOUD_REGIONS["default"] == "https://api.cloud.llamaindex.ai"
        assert LLAMACLOUD_REGIONS["eu"] == "https://api.cloud.eu.llamaindex.ai"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_llamacloud_base_url_no_region(self):
        """Test get_llamacloud_base_url with no region set (defaults to North America)."""
        result = get_llamacloud_base_url()
        assert result == "https://api.cloud.llamaindex.ai"

    @patch.dict(os.environ, {"LLAMACLOUD_REGION": "eu"})
    def test_get_llamacloud_base_url_eu_region(self):
        """Test get_llamacloud_base_url with EU region."""
        result = get_llamacloud_base_url()
        assert result == "https://api.cloud.eu.llamaindex.ai"

    @patch.dict(os.environ, {"LLAMACLOUD_REGION": "default"})
    def test_get_llamacloud_base_url_default_region(self):
        """Test get_llamacloud_base_url with default region."""
        result = get_llamacloud_base_url()
        assert result == "https://api.cloud.llamaindex.ai"

    @patch.dict(os.environ, {"LLAMACLOUD_REGION": "DEFAULT"})
    def test_get_llamacloud_base_url_case_insensitive(self):
        """Test get_llamacloud_base_url with case insensitive region."""
        result = get_llamacloud_base_url()
        assert result == "https://api.cloud.llamaindex.ai"

    @patch.dict(os.environ, {"LLAMACLOUD_REGION": "  default  "})
    def test_get_llamacloud_base_url_strips_whitespace(self):
        """Test get_llamacloud_base_url strips whitespace from region."""
        result = get_llamacloud_base_url()
        assert result == "https://api.cloud.llamaindex.ai"

    @patch.dict(os.environ, {"LLAMACLOUD_BASE_URL": "https://custom.api.com"})
    def test_get_llamacloud_base_url_custom_override(self):
        """Test get_llamacloud_base_url with custom base URL override."""
        result = get_llamacloud_base_url()
        assert result == "https://custom.api.com"

    @patch.dict(
        os.environ,
        {"LLAMACLOUD_BASE_URL": "https://custom.api.com", "LLAMACLOUD_REGION": "eu"},
    )
    def test_get_llamacloud_base_url_custom_override_precedence(self):
        """Test that custom base URL takes precedence over region."""
        result = get_llamacloud_base_url()
        assert result == "https://custom.api.com"

    @patch.dict(os.environ, {"LLAMACLOUD_REGION": "invalid"})
    def test_get_llamacloud_base_url_invalid_region(self):
        """Test get_llamacloud_base_url with invalid region raises error."""
        with pytest.raises(LlamaCloudConfigError) as exc_info:
            get_llamacloud_base_url()
        assert "Invalid LLAMACLOUD_REGION 'invalid'" in str(exc_info.value)
        assert "default, eu" in str(exc_info.value)

    @patch.dict(os.environ, {"LLAMACLOUD_API_KEY": "test-key"}, clear=True)
    def test_get_llamacloud_config_valid(self):
        """Test get_llamacloud_config with valid API key (defaults to North America)."""
        result = get_llamacloud_config()
        expected = {"token": "test-key", "base_url": "https://api.cloud.llamaindex.ai"}
        assert result == expected

    @patch.dict(
        os.environ,
        {"LLAMACLOUD_API_KEY": "test-key", "LLAMACLOUD_REGION": "eu"},
        clear=True,
    )
    def test_get_llamacloud_config_with_region(self):
        """Test get_llamacloud_config with region."""
        result = get_llamacloud_config()
        expected = {
            "token": "test-key",
            "base_url": "https://api.cloud.eu.llamaindex.ai",
        }
        assert result == expected

    @patch.dict(os.environ, {}, clear=True)
    def test_get_llamacloud_config_missing_api_key(self):
        """Test get_llamacloud_config with missing API key raises error."""
        with pytest.raises(LlamaCloudConfigError):
            get_llamacloud_config()

    @patch.dict(
        os.environ,
        {"LLAMACLOUD_API_KEY": "test-key", "LLAMACLOUD_REGION": "invalid"},
        clear=True,
    )
    def test_get_llamacloud_config_invalid_region(self):
        """Test get_llamacloud_config with invalid region raises error."""
        with pytest.raises(LlamaCloudConfigError):
            get_llamacloud_config()

    @patch("src.notebookllama.utils.AsyncLlamaCloud")
    @patch.dict(os.environ, {"LLAMACLOUD_API_KEY": "test-key"}, clear=True)
    def test_create_llamacloud_client_valid(self, mock_client_class):
        """Test create_llamacloud_client with valid configuration (defaults to North America)."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        result = create_llamacloud_client()

        mock_client_class.assert_called_once_with(
            token="test-key", base_url="https://api.cloud.llamaindex.ai"
        )
        assert result == mock_instance

    @patch("src.notebookllama.utils.AsyncLlamaCloud")
    @patch.dict(
        os.environ,
        {"LLAMACLOUD_API_KEY": "test-key", "LLAMACLOUD_REGION": "eu"},
        clear=True,
    )
    def test_create_llamacloud_client_with_region(self, mock_client_class):
        """Test create_llamacloud_client with region."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        result = create_llamacloud_client()

        mock_client_class.assert_called_once_with(
            token="test-key", base_url="https://api.cloud.eu.llamaindex.ai"
        )
        assert result == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    def test_create_llamacloud_client_missing_api_key(self):
        """Test create_llamacloud_client with missing API key raises error."""
        with pytest.raises(LlamaCloudConfigError):
            create_llamacloud_client()

    @patch("src.notebookllama.utils.LlamaExtract")
    @patch.dict(os.environ, {"LLAMACLOUD_API_KEY": "test-key"}, clear=True)
    def test_create_llama_extract_client_valid(self, mock_extract_class):
        """Test create_llama_extract_client with valid configuration (defaults to North America)."""
        mock_instance = MagicMock()
        mock_extract_class.return_value = mock_instance

        result = create_llama_extract_client()

        mock_extract_class.assert_called_once_with(
            api_key="test-key", base_url="https://api.cloud.llamaindex.ai"
        )
        assert result == mock_instance

    @patch("src.notebookllama.utils.LlamaExtract")
    @patch.dict(
        os.environ,
        {"LLAMACLOUD_API_KEY": "test-key", "LLAMACLOUD_REGION": "eu"},
        clear=True,
    )
    def test_create_llama_extract_client_with_region(self, mock_extract_class):
        """Test create_llama_extract_client with region."""
        mock_instance = MagicMock()
        mock_extract_class.return_value = mock_instance

        result = create_llama_extract_client()

        mock_extract_class.assert_called_once_with(
            api_key="test-key", base_url="https://api.cloud.eu.llamaindex.ai"
        )
        assert result == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    def test_create_llama_extract_client_missing_api_key(self):
        """Test create_llama_extract_client with missing API key raises error."""
        with pytest.raises(LlamaCloudConfigError):
            create_llama_extract_client()

    @patch("src.notebookllama.utils.LlamaParse")
    @patch.dict(os.environ, {"LLAMACLOUD_API_KEY": "test-key"}, clear=True)
    def test_create_llama_parse_client_default(self, mock_parse_class):
        """Test create_llama_parse_client with default parameters (defaults to North America)."""
        mock_instance = MagicMock()
        mock_parse_class.return_value = mock_instance

        result = create_llama_parse_client()

        mock_parse_class.assert_called_once_with(
            api_key="test-key",
            result_type="markdown",
            base_url="https://api.cloud.llamaindex.ai",
        )
        assert result == mock_instance

    @patch("src.notebookllama.utils.LlamaParse")
    @patch.dict(os.environ, {"LLAMACLOUD_API_KEY": "test-key"}, clear=True)
    def test_create_llama_parse_client_custom_result_type(self, mock_parse_class):
        """Test create_llama_parse_client with custom result type (defaults to North America)."""
        mock_instance = MagicMock()
        mock_parse_class.return_value = mock_instance

        result = create_llama_parse_client(result_type="text")

        mock_parse_class.assert_called_once_with(
            api_key="test-key",
            result_type="text",
            base_url="https://api.cloud.llamaindex.ai",
        )
        assert result == mock_instance

    @patch("src.notebookllama.utils.LlamaParse")
    @patch.dict(
        os.environ,
        {"LLAMACLOUD_API_KEY": "test-key", "LLAMACLOUD_REGION": "eu"},
        clear=True,
    )
    def test_create_llama_parse_client_with_region(self, mock_parse_class):
        """Test create_llama_parse_client with region."""
        mock_instance = MagicMock()
        mock_parse_class.return_value = mock_instance

        result = create_llama_parse_client()

        mock_parse_class.assert_called_once_with(
            api_key="test-key",
            result_type="markdown",
            base_url="https://api.cloud.eu.llamaindex.ai",
        )
        assert result == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    def test_create_llama_parse_client_missing_api_key(self):
        """Test create_llama_parse_client with missing API key raises error."""
        with pytest.raises(LlamaCloudConfigError):
            create_llama_parse_client()

    @patch("src.notebookllama.utils.LlamaCloudIndex")
    @patch.dict(os.environ, {}, clear=True)
    def test_create_llamacloud_index_valid(self, mock_index_class):
        """Test create_llamacloud_index with valid parameters (defaults to North America)."""
        mock_instance = MagicMock()
        mock_index_class.return_value = mock_instance

        result = create_llamacloud_index("test-key", "test-pipeline")

        mock_index_class.assert_called_once_with(
            api_key="test-key",
            pipeline_id="test-pipeline",
            base_url="https://api.cloud.llamaindex.ai",
        )
        assert result == mock_instance

    @patch("src.notebookllama.utils.LlamaCloudIndex")
    @patch.dict(os.environ, {"LLAMACLOUD_REGION": "eu"}, clear=True)
    def test_create_llamacloud_index_with_region(self, mock_index_class):
        """Test create_llamacloud_index with region."""
        mock_instance = MagicMock()
        mock_index_class.return_value = mock_instance

        result = create_llamacloud_index("test-key", "test-pipeline")

        mock_index_class.assert_called_once_with(
            api_key="test-key",
            pipeline_id="test-pipeline",
            base_url="https://api.cloud.eu.llamaindex.ai",
        )
        assert result == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    def test_create_llamacloud_index_missing_api_key(self):
        """Test create_llamacloud_index with missing API key raises error."""
        with pytest.raises(LlamaCloudConfigError) as exc_info:
            create_llamacloud_index("", "test-pipeline")
        assert "API key is required" in str(exc_info.value)

    @patch.dict(os.environ, {}, clear=True)
    def test_create_llamacloud_index_missing_pipeline_id(self):
        """Test create_llamacloud_index with missing pipeline ID raises error."""
        with pytest.raises(LlamaCloudConfigError) as exc_info:
            create_llamacloud_index("test-key", "")
        assert "Pipeline ID is required" in str(exc_info.value)

    @patch.dict(os.environ, {"LLAMACLOUD_REGION": "invalid"}, clear=True)
    def test_create_llamacloud_index_invalid_region(self):
        """Test create_llamacloud_index with invalid region raises error."""
        with pytest.raises(LlamaCloudConfigError):
            create_llamacloud_index("test-key", "test-pipeline")
