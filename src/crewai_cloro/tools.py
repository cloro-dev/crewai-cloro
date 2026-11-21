"""CrewAI tools for cloro: live Google and Bing results and AI search engine answers."""

import os
from typing import Any, ClassVar, Literal
from urllib.parse import quote

import httpx
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr

CLORO_API_BASE = "https://api.cloro.dev"
USER_AGENT = "crewai-cloro"
API_KEY_ENV = EnvVar(
    name="CLORO_API_KEY", description="API key for cloro (https://dashboard.cloro.dev).", required=True
)

# Bing Search and Bing AI Mode refuse a query longer than this after URL encoding.
BING_PROMPT_LIMIT = 3500

Engine = Literal["google_ai_mode", "bing_ai_mode", "chatgpt", "gemini"]

_ENGINES: dict[str, tuple[str, str]] = {
    "google_ai_mode": ("/v1/monitor/aimode", "Google AI Mode"),
    "bing_ai_mode": ("/v1/monitor/bing/aimode", "Bing AI Mode (Copilot Search in Bing)"),
    "chatgpt": ("/v1/monitor/chatgpt", "ChatGPT"),
    "gemini": ("/v1/monitor/gemini", "Gemini"),
}


class CloroError(RuntimeError):
    """The cloro API returned an error or a response that cannot be read."""


def _is_direct_link(url: Any) -> bool:
    """Return True for an http(s) link that is not a Google redirect or search page."""
    return (
        isinstance(url, str)
        and url.startswith(("http://", "https://"))
        and "google.com/goto" not in url
        and "google.com/search" not in url
    )


def _encoded_length(prompt: str) -> int:
    return len(quote(prompt.strip(), safe="-_.!~*'()").replace("%20", "+"))


def _sources(items: Any) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for item in items or []:
        url = item.get("url") if isinstance(item, dict) else None
        if _is_direct_link(url) and url not in [u for _, u in sources]:
            sources.append((item.get("label") or "", url))
    return sources


def _format_sources(sources: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {title}: {url}" if title else f"- {url}" for title, url in sources)


class _CloroTool(BaseTool):
    api_key: str | None = Field(default=None, description="cloro API key. Defaults to CLORO_API_KEY.")
    country: str = Field(default="US", description="ISO 3166-1 alpha-2 country code for the search location.")
    timeout: float = Field(default=180.0, description="Request timeout in seconds.")
    env_vars: list[EnvVar] = [API_KEY_ENV]
    _key: str = PrivateAttr(default="")

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        key = self.api_key or os.environ.get("CLORO_API_KEY")
        if not key:
            raise ValueError("Set CLORO_API_KEY or pass api_key. Get a key at https://dashboard.cloro.dev")
        self._key = key

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{CLORO_API_BASE}{path}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise CloroError(f"Request to cloro {path} failed: {error}") from error
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.is_error or not isinstance(data, dict) or not data.get("success"):
            error = data.get("error") if isinstance(data, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            raise CloroError(
                f"cloro {path} returned HTTP {response.status_code}: {code or 'no error code'} {message or ''}".strip()
            )
        result = data.get("result")
        if not isinstance(result, dict):
            raise CloroError(f"cloro {path} returned no result.")
        return result


class CloroGoogleSearchInput(BaseModel):
    """Input schema for CloroGoogleSearchTool."""

    query: str = Field(..., description="The search query.")
    country: str | None = Field(default=None, description="Optional ISO 3166-1 alpha-2 country code, for example GB.")


class CloroGoogleSearchTool(_CloroTool):
    """Live Google organic results, and optionally the Google AI Overview, through cloro."""

    name: str = "cloro_google_search"
    description: str = (
        "Search Google live and return the organic results of the first results page: title, URL and snippet. "
        "Use it for current facts, sources and what ranks on Google for a query."
    )
    args_schema: type[BaseModel] = CloroGoogleSearchInput
    path: ClassVar[str] = "/v1/monitor/google"
    label: ClassVar[str] = "Google"
    max_results: int = Field(default=10, description="The maximum number of organic results to return.")
    include_ai_overview: bool = Field(
        default=False, description="Also return the Google AI Overview and its sources. Costs 2 more credits."
    )

    def _run(self, query: str, country: str | None = None) -> str:
        payload: dict[str, Any] = {"query": query, "country": (country or self.country).upper()}
        if self.include_ai_overview:
            payload["include"] = {"aioverview": True}
        result = self._post(self.path, payload)

        parts: list[str] = []
        overview = result.get("aioverview")
        if self.include_ai_overview and isinstance(overview, dict) and overview.get("text"):
            text = f"{self.label} AI Overview:\n{overview['text']}"
            sources = _sources(overview.get("sources"))
            if sources:
                text += "\nSources:\n" + _format_sources(sources)
            parts.append(text)

        organic = [
            item
            for item in result.get("organicResults") or []
            if isinstance(item, dict) and _is_direct_link(item.get("link"))
        ][: self.max_results]
        for index, item in enumerate(organic, start=1):
            parts.append(f"{index}. {item.get('title') or ''}\n{item['link']}\n{item.get('snippet') or ''}".strip())
        return "\n\n".join(parts) if parts else f"No {self.label} results for: {query}"


class CloroBingSearchTool(CloroGoogleSearchTool):
    """Live Bing organic results, and optionally the Bing AI Overview, through cloro."""

    name: str = "cloro_bing_search"
    description: str = (
        "Search Bing live and return the organic results of the first results page: title, URL and snippet. "
        "Use it for current facts, sources and what ranks on Bing for a query."
    )
    path: ClassVar[str] = "/v1/monitor/bing"
    label: ClassVar[str] = "Bing"
    include_ai_overview: bool = Field(
        default=False, description="Also return the Bing AI Overview and its sources. Costs 2 more credits."
    )

    def _run(self, query: str, country: str | None = None) -> str:
        if _encoded_length(query) > BING_PROMPT_LIMIT:
            raise ValueError(
                "The query is too long for Bing Search: at most 3,500 characters after URL encoding. "
                "Non-Latin scripts allow fewer characters."
            )
        return super()._run(query, country)


class CloroAIAnswerInput(BaseModel):
    """Input schema for CloroAIAnswerTool."""

    prompt: str = Field(..., description="The question to ask the AI search engine.")
    country: str | None = Field(default=None, description="Optional ISO 3166-1 alpha-2 country code, for example GB.")


class CloroAIAnswerTool(_CloroTool):
    """The answer an AI search engine gives to a prompt, with the sources it cites, through cloro."""

    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] = CloroAIAnswerInput
    engine: Engine = Field(
        default="google_ai_mode",
        description="The AI search engine: google_ai_mode, bing_ai_mode, chatgpt or gemini.",
    )

    def model_post_init(self, context: Any) -> None:
        label = _ENGINES[self.engine][1]
        if not self.name:
            self.name = f"cloro_{self.engine}"
        if not self.description:
            self.description = (
                f"Ask {label} a question live and return the answer it shows users, as Markdown, "
                "with the sources it cites. Use it to see what this AI engine says about a topic or a brand."
            )
        super().model_post_init(context)

    def _run(self, prompt: str, country: str | None = None) -> str:
        path, label = _ENGINES[self.engine]
        if self.engine == "bing_ai_mode" and _encoded_length(prompt) > BING_PROMPT_LIMIT:
            raise ValueError(
                "The prompt is too long for Bing AI Mode: at most 3,500 characters after URL encoding. "
                "Non-Latin scripts allow fewer characters."
            )
        result = self._post(
            path, {"prompt": prompt, "country": (country or self.country).upper(), "include": {"markdown": True}}
        )
        answer = result.get("markdown") or result.get("text") or ""
        text = f"{label} answer:\n{answer}" if answer else f"{label} returned no answer for: {prompt}"
        sources = _sources(result.get("sources"))
        if sources:
            text += "\n\nSources:\n" + _format_sources(sources)
        return text
