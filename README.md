# crewai-cloro

[CrewAI](https://www.crewai.com) tools for [cloro](https://cloro.dev/?utm_source=crewai-cloro&utm_medium=byok). cloro
loads the live page of a search engine or an AI assistant for a query and a country, and returns the result as
structured data.

| Tool | cloro endpoint | The agent gets | Credits per call |
|---|---|---|--:|
| `CloroGoogleSearchTool` | `POST /v1/monitor/google` | Google organic results: title, URL and snippet. With `include_ai_overview=True`, the Google AI Overview and its sources first. | 5, or 7 with the AI Overview |
| `CloroAIAnswerTool(engine="google_ai_mode")` | `POST /v1/monitor/aimode` | The Google AI Mode answer as Markdown, with its sources | 6 |
| `CloroAIAnswerTool(engine="bing_ai_mode")` | `POST /v1/monitor/bing/aimode` | The Bing AI Mode (Copilot Search in Bing) answer, with its sources | 6 |
| `CloroAIAnswerTool(engine="chatgpt")` | `POST /v1/monitor/chatgpt` | The ChatGPT answer that users see, with its sources | 7 |
| `CloroAIAnswerTool(engine="gemini")` | `POST /v1/monitor/gemini` | The Gemini answer, with its sources | 6 |

## Installation

```bash
pip install crewai-cloro
```

Get an API key at [dashboard.cloro.dev](https://dashboard.cloro.dev/api-keys?utm_source=crewai-cloro&utm_medium=byok).
The free plan includes 500 credits each month. Set the key as `CLORO_API_KEY`, or pass `api_key=...` to a tool.

## Usage

```python
from crewai import Agent, Crew, Task
from crewai_cloro import CloroAIAnswerTool, CloroGoogleSearchTool

researcher = Agent(
    role="Search analyst",
    goal="Compare what Google and AI assistants say about a topic.",
    backstory="You check live sources and cite the URLs you use.",
    tools=[
        CloroGoogleSearchTool(max_results=5, include_ai_overview=True),
        CloroAIAnswerTool(engine="chatgpt"),
        CloroAIAnswerTool(engine="google_ai_mode"),
    ],
)

task = Task(
    description="Which SERP APIs do Google, ChatGPT and Google AI Mode recommend for AI agents?",
    expected_output="A short comparison with the cited URLs.",
    agent=researcher,
)

print(Crew(agents=[researcher], tasks=[task]).kickoff())
```

Each `CloroAIAnswerTool` gets its own tool name, `cloro_<engine>`, so one agent can hold several engines. Pass
`name=` to change it.

A direct call returns the same text the agent gets:

```python
print(CloroGoogleSearchTool().run(query="retrieval-augmented generation", country="GB"))
```

### Parameters

Both tools take:

- `api_key`: the cloro API key. Defaults to the `CLORO_API_KEY` environment variable.
- `country`: ISO 3166-1 alpha-2 country code for the search location (default `US`). The agent can also pass
  `country` per call.
- `timeout`: request timeout in seconds (default 180). An AI answer can take 15 to 60 seconds.

`CloroGoogleSearchTool` also takes `max_results` (default 10) and `include_ai_overview` (default `False`).
Results with an empty link or a Google redirect link are skipped.

`CloroAIAnswerTool` also takes `engine`: `google_ai_mode` (default), `bing_ai_mode`, `chatgpt` or `gemini`.

## Limits

- Bing AI Mode accepts a prompt of up to 3,500 characters after URL encoding. Non-Latin scripts allow fewer
  characters. The tool refuses a longer prompt before it sends the request.
- Your plan sets how many requests can run at the same time. A request sent while all slots are busy gets a `429`
  (`CONCURRENCY_LIMIT_EXCEEDED`).
- An API error raises `CloroError` with the HTTP status, the cloro error code and the message.

## Development

```bash
uv run --python 3.12 --with crewai --with httpx --with pytest -- env PYTHONPATH=src python -m pytest tests
```

The tests mock the network. The API reference is at
[cloro.dev/docs](https://cloro.dev/docs?utm_source=crewai-cloro&utm_medium=byok).

## License

MIT
