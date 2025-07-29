import os
import sys
from dotenv import load_dotenv
import pandas as pd
import json
import warnings
from datetime import datetime

from mrkdwn_analysis import MarkdownAnalyzer
from mrkdwn_analysis.markdown_analyzer import InlineParser, MarkdownParser
from llama_cloud_services.extract import SourceText
from typing_extensions import override
from typing import List, Tuple, Union, Optional, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from notebookllama.utils import (
    create_llamacloud_client,
    create_llama_extract_client,
    create_llama_parse_client,
)

load_dotenv()

if (
    os.getenv("LLAMACLOUD_API_KEY", None)
    and os.getenv("EXTRACT_AGENT_ID", None)
    and os.getenv("LLAMACLOUD_PIPELINE_ID", None)
):
    CLIENT = create_llamacloud_client()
    llama_extract_client = create_llama_extract_client()
    EXTRACT_AGENT = llama_extract_client.get_agent(id=os.getenv("EXTRACT_AGENT_ID"))
    PARSER = create_llama_parse_client(result_type="markdown")
    PIPELINE_ID = os.getenv("LLAMACLOUD_PIPELINE_ID")


class MarkdownTextAnalyzer(MarkdownAnalyzer):
    @override
    def __init__(self, text: str):
        self.text = text
        parser = MarkdownParser(self.text)
        self.tokens = parser.parse()
        self.references = parser.references
        self.footnotes = parser.footnotes
        self.inline_parser = InlineParser(
            references=self.references, footnotes=self.footnotes
        )
        self._parse_inline_tokens()


def md_table_to_pd_dataframe(md_table: Dict[str, list]) -> Optional[pd.DataFrame]:
    try:
        df = pd.DataFrame()
        for i in range(len(md_table["header"])):
            ls = [row[i] for row in md_table["rows"]]
            df[md_table["header"][i]] = ls
        return df
    except Exception as e:
        warnings.warn(f"Skipping table as an error occurred: {e}")
        return None


def rename_and_remove_past_images(path: str = "static/") -> List[str]:
    renamed = []
    if os.path.exists(path) and len(os.listdir(path)) >= 0:
        for image_file in os.listdir(path):
            image_path = os.path.join(path, image_file)
            if os.path.isfile(image_path) and "_at_" not in image_path:
                with open(image_path, "rb") as img:
                    bts = img.read()
                new_path = (
                    os.path.splitext(image_path)[0].replace("_current", "")
                    + f"_at_{datetime.now().strftime('%Y_%d_%m_%H_%M_%S_%f')[:-3]}.png"
                )
                with open(
                    new_path,
                    "wb",
                ) as img_tw:
                    img_tw.write(bts)
                renamed.append(new_path)
                os.remove(image_path)
    return renamed


def rename_and_remove_current_images(images: List[str]) -> List[str]:
    imgs = []
    for image in images:
        with open(image, "rb") as rb:
            bts = rb.read()
        with open(os.path.splitext(image)[0] + "_current.png", "wb") as wb:
            wb.write(bts)
        imgs.append(os.path.splitext(image)[0] + "_current.png")
        os.remove(image)
    return imgs


async def parse_file(
    file_path: str, with_images: bool = False, with_tables: bool = False
) -> Union[Tuple[Optional[str], Optional[List[str]], Optional[List[pd.DataFrame]]]]:
    images: Optional[List[str]] = None
    text: Optional[str] = None
    tables: Optional[List[pd.DataFrame]] = None
    document = await PARSER.aparse(file_path=file_path)
    md_content = await document.aget_markdown_documents()
    if len(md_content) != 0:
        text = "\n\n---\n\n".join([doc.text for doc in md_content])
    if with_images:
        rename_and_remove_past_images()
        imgs = await document.asave_all_images("static/")
        images = rename_and_remove_current_images(imgs)
    if with_tables:
        if text is not None:
            analyzer = MarkdownTextAnalyzer(text)
            md_tables = analyzer.identify_tables()["Table"]
            tables = []
            for md_table in md_tables:
                table = md_table_to_pd_dataframe(md_table=md_table)
                if table is not None:
                    tables.append(table)
                    os.makedirs("data/extracted_tables/", exist_ok=True)
                    table.to_csv(
                        f"data/extracted_tables/table_{datetime.now().strftime('%Y_%d_%m_%H_%M_%S_%f')[:-3]}.csv",
                        index=False,
                    )
    return text, images, tables


async def process_file(
    filename: str,
) -> Union[Tuple[str, None], Tuple[None, None], Tuple[str, str]]:
    with open(filename, "rb") as f:
        file = await CLIENT.files.upload_file(upload_file=f)
    files = [{"file_id": file.id}]
    await CLIENT.pipelines.add_files_to_pipeline_api(
        pipeline_id=PIPELINE_ID, request=files
    )
    text, _, _ = await parse_file(file_path=filename)
    if text is None:
        return None, None
    extraction_output = await EXTRACT_AGENT.aextract(
        files=SourceText(text_content=text, filename=file.name)
    )
    if extraction_output:
        return json.dumps(extraction_output.data, indent=4), text
    return None, None


async def get_plots_and_tables(
    file_path: str,
) -> Union[Tuple[Optional[List[str]], Optional[List[pd.DataFrame]]]]:
    _, images, tables = await parse_file(
        file_path=file_path, with_images=True, with_tables=True
    )
    return images, tables
