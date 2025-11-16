# due-diligence

A collection of scripts and small utilities for automating due-diligence research, data collection, and professional report drafting. This repository contains a set of tools that scrape and aggregate news and financial data, create draft reports and emails, and include experiments integrating different LLM/agent implementations.

## Contents

- `autogen_implementation.py` — experiment with Autogen-style orchestration (project-specific integration).
- `crewai_implementation.py` — CreAI integration experiments.
- `langgraph_implementation.py` — LangGraph-related scripts and usage examples.
- `llama_index_implementastion.py` — LlamaIndex experiments (note: filename contains a minor spelling variant).
- `mcp_server.py` — lightweight server or entrypoint for MCP-related workflows.
- `medium.py` — utilities for formatting or publishing content (project-specific).
- `test_crew.py` — simple test / example runner for core functionality.
- `web_search_news.py` — news/web search utilities used to collect recent articles.
- `credit_risk.db` — a small local SQLite database used by some data-processing scripts (do not commit secrets into DB).
- `tools/` — helper scripts used by the project:
	- `tools/email_calendar.py` — calendar/email utilities.
	- `tools/email_drafter.py` — helper for drafting professional emails.
	- `tools/professional_report.docx` — example report template (binary document).
	- `tools/report_creator.py` — report assembly utilities.
	- `tools/stock_data.py` — stock data retrieval helpers.
	- `tools/web_search_news.py` — duplicate/alternate web search helper (careful which one you run).

## Quick start

Prerequisites
- Python 3.10+ (project was developed in a Python 3.12 environment — see `.pyc` files in the repo).
- Git (to clone the repo)

1. Clone the repository:

```bash
git clone <repo-url>
cd due-diligence
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# If you have a requirements.txt, install dependencies:
# pip install -r requirements.txt
```

3. Run a script (examples):

```bash
python3 web_search_news.py
python3 tools/report_creator.py
python3 test_crew.py
```

Note: Some scripts expect environment configuration, API keys, or a populated `credit_risk.db`. Inspect the top of each script for configuration notes and required environment variables.

## Usage and examples

- Use `web_search_news.py` (or `tools/web_search_news.py`) to collect recent news. Check each file for flags/options.
- Use `tools/stock_data.py` to fetch equity data used by due-diligence flows.
- Use `tools/email_drafter.py` and `tools/report_creator.py` to assemble draft communications and reports.
- `mcp_server.py` is an entrypoint that may provide a local API or orchestration layer — inspect the file for runtime flags and ports.

## Project notes & recommendations

- Add a `requirements.txt` or `pyproject.toml` to pin dependencies used by the scripts.
- Add a LICENSE file if you plan to publish or share the project publicly.
- Consider adding small unit tests (pytest) and a CI workflow to validate core scripts.

## Contributing

1. Open an issue describing the change or improvement.
2. Create a short-lived branch and make incremental commits.
3. Submit a pull request with a description of the change and any verification steps.

## Running tests

This repository includes `test_crew.py` as a lightweight test/example runner. Execute it directly:

```bash
python3 test_crew.py
```

For a more formal test setup, add `pytest` and create tests under `tests/`.

## Contact

If you need context about how a script is used or where data comes from, inspect the top-level scripts and the `tools/` directory or reach out to the repository owner.

---

Generated: November 16, 2025
