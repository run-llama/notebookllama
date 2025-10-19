import streamlit as st
import io
import os
import asyncio
import tempfile as temp
from dotenv import load_dotenv
import sys
import time
import randomname
import streamlit.components.v1 as components

from pathlib import Path
from documents import ManagedDocument, DocumentManager
from audio import PODCAST_GEN, PodcastConfig
from typing import Tuple
from workflow import NotebookLMWorkflow, FileInputEvent, NotebookOutputEvent
from instrumentation import OtelTracesSqlEngine
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)

load_dotenv()

# define a custom span exporter
span_exporter = OTLPSpanExporter("http://localhost:4318/v1/traces")

# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry(
    service_name_or_resource="agent.traces",
    span_exporter=span_exporter,
    debug=True,
)
engine_url = f"postgresql+psycopg2://{os.getenv('pgql_user')}:{os.getenv('pgql_psw')}@localhost:5432/{os.getenv('pgql_db')}"
sql_engine = OtelTracesSqlEngine(
    engine_url=engine_url,
    table_name="agent_traces",
    service_name="agent.traces",
)
document_manager = DocumentManager(engine_url=engine_url)

WF = NotebookLMWorkflow(timeout=600)


# Read the HTML file
def read_html_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


async def run_workflow(
    file: io.BytesIO, document_title: str
) -> Tuple[str, str, str, str, str]:
    # Create temp file with proper Windows handling
    with temp.NamedTemporaryFile(suffix=".pdf", delete=False) as fl:
        content = file.getvalue()
        fl.write(content)
        fl.flush()  # Ensure data is written
        temp_path = fl.name

    try:
        st_time = int(time.time() * 1000000)
        ev = FileInputEvent(file=temp_path)
        result: NotebookOutputEvent = await WF.run(start_event=ev)

        q_and_a = ""
        for q, a in zip(result.questions, result.answers):
            q_and_a += f"**{q}**\n\n{a}\n\n"
        bullet_points = "## Bullet Points\n\n- " + "\n- ".join(result.highlights)

        mind_map = result.mind_map
        if Path(mind_map).is_file():
            mind_map = read_html_file(mind_map)
            try:
                os.remove(result.mind_map)
            except OSError:
                pass  # File might be locked on Windows

        end_time = int(time.time() * 1000000)
        sql_engine.to_sql_database(start_time=st_time, end_time=end_time)
        document_manager.put_documents(
            [
                ManagedDocument(
                    document_name=document_title,
                    content=result.md_content,
                    summary=result.summary,
                    q_and_a=q_and_a,
                    mindmap=mind_map,
                    bullet_points=bullet_points,
                )
            ]
        )
        return result.md_content, result.summary, q_and_a, bullet_points, mind_map

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            await asyncio.sleep(0.1)
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Give up if still locked


def sync_run_workflow(file: io.BytesIO, document_title: str):
    try:
        # Try to use existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, schedule the coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, run_workflow(file, document_title)
                )
                return future.result()
        else:
            return loop.run_until_complete(run_workflow(file, document_title))
    except RuntimeError:
        # No event loop exists, create one
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        return asyncio.run(run_workflow(file, document_title))


async def create_podcast(file_content: str, config: PodcastConfig = None):
    audio_fl = await PODCAST_GEN.create_conversation(
        file_transcript=file_content, config=config
    )
    return audio_fl


def sync_create_podcast(file_content: str, config: PodcastConfig = None):
    return asyncio.run(create_podcast(file_content=file_content, config=config))


# Display the network
st.set_page_config(
    page_title="NotebookLlaMa - Home",
    page_icon="🏠",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/run-llama/notebooklm-clone/discussions/categories/general",
        "Report a bug": "https://github.com/run-llama/notebooklm-clone/issues/",
        "About": "An OSS alternative to NotebookLM that runs with the power of a flully Llama!",
    },
)
st.sidebar.header("Home🏠")
st.sidebar.info("To switch to the Document Chat, select it from above!🔺")
st.markdown("---")
st.markdown("## NotebookLlaMa - Home🦙")

# Initialize session state BEFORE creating the text input
if "workflow_results" not in st.session_state:
    st.session_state.workflow_results = None
if "document_title" not in st.session_state:
    st.session_state.document_title = randomname.get_name(
        adj=("music_theory", "geometry", "emotions"), noun=("cats", "food")
    )

# Use session_state as the value and update it when changed
document_title = st.text_input(
    label="Document Title",
    value=st.session_state.document_title,
    key="document_title_input",
)

# Update session state when the input changes
if document_title != st.session_state.document_title:
    st.session_state.document_title = document_title

file_input = st.file_uploader(
    label="Upload your source PDF file!", accept_multiple_files=False
)

if file_input is not None:
    # First button: Process Document
    if st.button("Process Document", type="primary"):
        with st.spinner("Processing document... This may take a few minutes."):
            try:
                md_content, summary, q_and_a, bullet_points, mind_map = (
                    sync_run_workflow(file_input, st.session_state.document_title)
                )
                st.session_state.workflow_results = {
                    "md_content": md_content,
                    "summary": summary,
                    "q_and_a": q_and_a,
                    "bullet_points": bullet_points,
                    "mind_map": mind_map,
                }
                st.success("Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    # Display results if available
    if st.session_state.workflow_results:
        results = st.session_state.workflow_results

        # Summary
        st.markdown("## Summary")
        st.markdown(results["summary"])

        # Bullet Points
        st.markdown(results["bullet_points"])

        # FAQ (toggled)
        with st.expander("FAQ"):
            st.markdown(results["q_and_a"])

        # Mind Map
        if results["mind_map"]:
            st.markdown("## Mind Map")
            components.html(results["mind_map"], height=800, scrolling=True)

        # Podcast Configuration Panel
        st.markdown("---")
        st.markdown("## Podcast Configuration")

        with st.expander("Customize Your Podcast", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                style = st.selectbox(
                    "Conversation Style",
                    ["conversational", "interview", "debate", "educational"],
                    help="The overall style of the podcast conversation",
                )

                tone = st.selectbox(
                    "Tone",
                    ["friendly", "professional", "casual", "energetic"],
                    help="The tone of voice for the conversation",
                )

                target_audience = st.selectbox(
                    "Target Audience",
                    ["general", "technical", "business", "expert", "beginner"],
                    help="Who is the intended audience for this podcast?",
                )

            with col2:
                speaker1_role = st.text_input(
                    "Speaker 1 Role",
                    value="host",
                    help="The role or persona of the first speaker",
                )

                speaker2_role = st.text_input(
                    "Speaker 2 Role",
                    value="guest",
                    help="The role or persona of the second speaker",
                )

            # Focus Topics
            st.markdown("**Focus Topics** (optional)")
            focus_topics_input = st.text_area(
                "Enter topics to emphasize (one per line)",
                help="List specific topics you want the podcast to focus on. Leave empty for general coverage.",
                placeholder="How can this be applied for Machine Learning Applications?\nUnderstand the historical context\nFuture Implications",
            )

            # Parse focus topics
            focus_topics = None
            if focus_topics_input.strip():
                focus_topics = [
                    topic.strip()
                    for topic in focus_topics_input.split("\n")
                    if topic.strip()
                ]

            # Custom Prompt
            custom_prompt = st.text_area(
                "Custom Instructions (optional)",
                help="Add any additional instructions for the podcast generation",
                placeholder="Make sure to explain technical concepts simply and include real-world examples...",
            )

            # Create config object
            podcast_config = PodcastConfig(
                style=style,
                tone=tone,
                focus_topics=focus_topics,
                target_audience=target_audience,
                custom_prompt=custom_prompt if custom_prompt.strip() else None,
                speaker1_role=speaker1_role,
                speaker2_role=speaker2_role,
            )

        # Second button: Generate Podcast
        if st.button("Generate In-Depth Conversation", type="secondary"):
            with st.spinner("Generating podcast... This may take several minutes."):
                try:
                    audio_file = sync_create_podcast(
                        results["md_content"], config=podcast_config
                    )
                    st.success("Podcast generated successfully!")

                    # Display audio player
                    st.markdown("## Generated Podcast")
                    if os.path.exists(audio_file):
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        os.remove(audio_file)
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.error("Audio file not found.")

                except Exception as e:
                    st.error(f"Error generating podcast: {str(e)}")

else:
    st.info("Please upload a PDF file to get started.")

if __name__ == "__main__":
    instrumentor.start_registering()

# Add a fun personalized footer and dynamic greeting
import datetime

current_hour = datetime.datetime.now().hour
if current_hour < 12:
    greeting = "Good morning ☀️"
elif current_hour < 18:
    greeting = "Good afternoon 🌤️"
else:
    greeting = "Good evening 🌙"

footer_quotes = [
    "“Simplicity is the soul of efficiency.” – Austin Freeman",
    "“Code is like humor. When you have to explain it, it’s bad.” – Cory House",
    "“Programs must be written for people to read.” – Harold Abelson",
    "“First, solve the problem. Then, write the code.” – John Johnson",
]
st.markdown("---")
st.markdown(f"### {greeting}, curious mind!")
st.caption(random.choice(footer_quotes))
