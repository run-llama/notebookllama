# NotebookLlaMa🦙

## A fluffy and open-source alternative to NotebookLM!

https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2

<p align="center">
  A fully open-source alternative to NotebookLM, backed by <a href="https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM"><strong>LlamaCloud</strong></a>.
</p>

<p align="center">
    <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
    <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
    <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
    <br>
    <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

### Prerequisites

This project uses `uv` to manage dependencies. Before you begin, make sure you have `uv` installed.

On macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more install options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

---

### Get it up and running!

**1. Clone the Repository**

```bash
git clone https://github.com/run-llama/notebookllama
cd notebookllama/
```

**2. Install Dependencies**

```bash
uv sync
```

**3. Configure API Keys**

First, create your `.env` file by renaming the example file:

```bash
mv .env.example .env
```

Next, open the `.env` file and add your API keys:

- `OPENAI_API_KEY`: find it [on OpenAI Platform](https://platform.openai.com/api-keys)
- `ELEVENLABS_API_KEY`: find it [on ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
- `LLAMACLOUD_API_KEY`: find it [on LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

> **🌍 Regional Support**: LlamaCloud operates in multiple regions. If you're using a European region, configure it in your `.env` file:
>
> - For **North America**: This is the default region - no configuration necesary.
> - For **Europe (EU)**: Uncomment and set `LLAMACLOUD_REGION="eu"`

**4. Activate the Virtual Environment**

(on mac/unix)

```bash
source .venv/bin/activate
```

(on Windows):

```bash
.\.venv\Scripts\activate
```

**5. Create LlamaCloud Agent & Pipeline**

You will now execute two scripts to configure your backend agents and pipelines.

First, create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Next, run the interactive setup wizard to configure your index pipeline.

> **⚡ Quick Start (Default OpenAI):**
> For the fastest setup, select **"With Default Settings"** when prompted. This will automatically create a pipeline using OpenAI's `text-embedding-3-small` embedding model.

> **🧠 Advanced (Custom Embedding Models):**
> To use a different embedding model, select **"With Custom Settings"** and follow the on-screen instructions.

Run the wizard with the following command:

```bash
uv run tools/create_llama_cloud_index.py
```

**6. Launch Backend Services**

This command will start the required Postgres and Jaeger containers.

```bash
docker compose up -d
```

**7. Run the Application**

First, run the **MCP** server:

```bash
uv run src/notebookllama/server.py
```

Then, in a **new terminal window**, launch the Streamlit app:

```bash
streamlit run src/notebookllama/Home.py
```

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already_

And start exploring the app at `http://localhost:8501/`.

---

### Contributing

Contribute to this project following the [guidelines](./CONTRIBUTING.md).

### License

This project is provided under an [MIT License](./LICENSE).
