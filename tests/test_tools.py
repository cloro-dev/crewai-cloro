from unittest.mock import patch

import httpx
import pytest
from crewai.tools import BaseTool

from crewai_cloro import CloroAIAnswerTool, CloroBingSearchTool, CloroError, CloroGoogleSearchTool


def _response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "https://api.cloro.dev"))


GOOGLE = {
    "success": True,
    "result": {
        "organicResults": [
            {"position": 1, "title": "First", "link": "https://example.com/a", "snippet": "A."},
            {"position": 2, "title": "Unresolved", "link": "", "snippet": "No link."},
            {"position": 3, "title": "Redirect", "link": "https://www.google.com/goto?x=1", "snippet": "R."},
            {"position": 4, "title": "Second", "link": "https://example.org/b", "snippet": "B."},
        ],
        "aioverview": {
            "text": "An overview.",
            "sources": [{"label": "Src", "url": "https://example.net"}],
        },
    },
}

ANSWER = {
    "success": True,
    "result": {
        "text": "Plain.",
        "markdown": "**Answer.**",
        "sources": [
            {"position": 1, "label": "Source", "url": "https://example.com"},
            {"position": 2, "label": "Again", "url": "https://example.com"},
            {"position": 3, "label": "Search page", "url": "https://www.google.com/search?q=x"},
        ],
    },
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("CLORO_API_KEY", "env-key")


def test_tools_are_crewai_tools():
    assert isinstance(CloroGoogleSearchTool(), BaseTool)
    assert isinstance(CloroAIAnswerTool(), BaseTool)
    assert [e.name for e in CloroGoogleSearchTool().env_vars] == ["CLORO_API_KEY"]


def test_missing_key(monkeypatch):
    monkeypatch.delenv("CLORO_API_KEY")
    with pytest.raises(ValueError, match="CLORO_API_KEY"):
        CloroGoogleSearchTool()


def test_explicit_key_wins():
    with patch("crewai_cloro.tools.httpx.post", return_value=_response(200, GOOGLE)) as post:
        CloroGoogleSearchTool(api_key="k").run(query="q")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer k"


@patch("crewai_cloro.tools.httpx.post")
def test_google_search(post):
    post.return_value = _response(200, GOOGLE)
    out = CloroGoogleSearchTool(max_results=5).run(query="serp api", country="de")

    assert post.call_args.args[0] == "https://api.cloro.dev/v1/monitor/google"
    assert post.call_args.kwargs["json"] == {"query": "serp api", "country": "DE"}
    assert out == "1. First\nhttps://example.com/a\nA.\n\n2. Second\nhttps://example.org/b\nB."


@patch("crewai_cloro.tools.httpx.post")
def test_google_search_with_ai_overview(post):
    post.return_value = _response(200, GOOGLE)
    out = CloroGoogleSearchTool(include_ai_overview=True, max_results=1).run(query="q")

    assert post.call_args.kwargs["json"] == {"query": "q", "country": "US", "include": {"aioverview": True}}
    assert out.startswith("Google AI Overview:\nAn overview.\nSources:\n- Src: https://example.net\n\n1. First")
    assert "Second" not in out


@patch("crewai_cloro.tools.httpx.post")
def test_google_search_no_results(post):
    post.return_value = _response(200, {"success": True, "result": {"organicResults": []}})
    assert CloroGoogleSearchTool().run(query="nothing") == "No Google results for: nothing"


@patch("crewai_cloro.tools.httpx.post")
def test_bing_search(post):
    post.return_value = _response(200, GOOGLE)
    tool = CloroBingSearchTool(max_results=5)
    out = tool.run(query="serp api", country="gb")

    assert tool.name == "cloro_bing_search"
    assert isinstance(tool, BaseTool)
    assert post.call_args.args[0] == "https://api.cloro.dev/v1/monitor/bing"
    assert post.call_args.kwargs["json"] == {"query": "serp api", "country": "GB"}
    assert out == "1. First\nhttps://example.com/a\nA.\n\n2. Second\nhttps://example.org/b\nB."


@patch("crewai_cloro.tools.httpx.post")
def test_bing_search_with_ai_overview(post):
    post.return_value = _response(200, GOOGLE)
    out = CloroBingSearchTool(include_ai_overview=True, max_results=1).run(query="q")

    assert post.call_args.kwargs["json"] == {"query": "q", "country": "US", "include": {"aioverview": True}}
    assert out.startswith("Bing AI Overview:\nAn overview.\nSources:\n- Src: https://example.net\n\n1. First")


@patch("crewai_cloro.tools.httpx.post")
def test_bing_search_no_results(post):
    post.return_value = _response(200, {"success": True, "result": {"organicResults": []}})
    assert CloroBingSearchTool().run(query="nothing") == "No Bing results for: nothing"


@patch("crewai_cloro.tools.httpx.post")
def test_bing_search_query_too_long(post):
    with pytest.raises(ValueError, match="too long for Bing Search"):
        CloroBingSearchTool()._run(query="中" * 600)
    post.assert_not_called()


@pytest.mark.parametrize(
    ("engine", "path", "label"),
    [
        ("google_ai_mode", "/v1/monitor/aimode", "Google AI Mode"),
        ("bing_ai_mode", "/v1/monitor/bing/aimode", "Bing AI Mode (Copilot Search in Bing)"),
        ("chatgpt", "/v1/monitor/chatgpt", "ChatGPT"),
        ("gemini", "/v1/monitor/gemini", "Gemini"),
    ],
)
@patch("crewai_cloro.tools.httpx.post")
def test_ai_answers(post, engine, path, label):
    post.return_value = _response(200, ANSWER)
    tool = CloroAIAnswerTool(engine=engine, country="gb")
    out = tool.run(prompt="What is a SERP?")

    assert tool.name == f"cloro_{engine}"
    assert label in tool.description
    assert post.call_args.args[0] == f"https://api.cloro.dev{path}"
    expected = {"prompt": "What is a SERP?", "country": "GB", "include": {"markdown": True}}
    assert post.call_args.kwargs["json"] == expected
    assert out == f"{label} answer:\n**Answer.**\n\nSources:\n- Source: https://example.com"


def test_custom_name_is_kept():
    assert CloroAIAnswerTool(engine="gemini", name="ask_gemini").name == "ask_gemini"


@patch("crewai_cloro.tools.httpx.post")
def test_bing_prompt_too_long(post):
    # 600 CJK characters are 5,400 characters once URL-encoded.
    with pytest.raises(ValueError, match="too long for Bing AI Mode"):
        CloroAIAnswerTool(engine="bing_ai_mode")._run(prompt="中" * 600)
    post.assert_not_called()


@patch("crewai_cloro.tools.httpx.post")
def test_api_error(post):
    post.return_value = _response(
        402, {"success": False, "error": {"code": "INSUFFICIENT_CREDITS", "message": "Not enough credits"}}
    )
    with pytest.raises(CloroError, match="402: INSUFFICIENT_CREDITS Not enough credits"):
        CloroAIAnswerTool(engine="gemini")._run(prompt="hi")


@patch("crewai_cloro.tools.httpx.post")
def test_network_error(post):
    post.side_effect = httpx.ConnectError("down")
    with pytest.raises(CloroError, match="failed"):
        CloroGoogleSearchTool()._run(query="hi")
