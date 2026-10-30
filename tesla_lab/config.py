"""Settings, read from environment / .env.

Mode:
  TESLA_LAB_MODE=mock   canned research + canned LLM answers, real Python math. No keys needed.
  TESLA_LAB_MODE=live   real Tavily + real Nemotron on Nebius Token Factory.
  TESLA_LAB_MODE=auto   (default) live if both keys are set, otherwise mock.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional, so mock mode works even before `pip install`
    from dotenv import load_dotenv

    load_dotenv(".env.local")
    load_dotenv(".env")
except Exception:  # fallback manual .env loader if python-dotenv is not installed
    pass

from pathlib import Path
for _fname in [".env.local", ".env"]:
    _env_path = Path(__file__).resolve().parents[1] / _fname
    if _env_path.exists():
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.split("#")[0].strip()
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v

# Verified 2026-10-12 at https://docs.tokenfactory.nebius.com/quickstart
NEBIUS_BASE_URL_DEFAULT = "https://api.tokenfactory.nebius.com/v1/"
# Named as live replacement models in Nebius' Aug-2026 deprecation notice
# (https://docs.tokenfactory.nebius.com/august-2026-deprecation-notice). Confirm with
# `python scripts/check_nebius.py`, which lists the models your key can actually call.
NEMOTRON_MODEL_DEFAULT = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_FAST_MODEL_DEFAULT = "nvidia/Nemotron-3_5-Lightning"
# From Nebius' image-generation example; NOT confirmed as currently served. Check the catalog.
IMAGE_MODEL_DEFAULT = "black-forest-labs/flux-schnell"
# Verified 2026-10-12 at https://docs.tavily.com/documentation/api-reference/endpoint/search
TAVILY_BASE_URL_DEFAULT = "https://api.tavily.com"
# v3 math checker. Model ids verified on Hugging Face 2026-10-13: nvidia/OpenMath-Nemotron-1.5B/7B/14B/32B
# (CC-BY-4.0). Self-hosted with vLLM (deploy/openmath/); there is no default URL.
OPENMATH_MODEL_DEFAULT = "nvidia/OpenMath-Nemotron-7B"


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    mode: str
    nebius_api_key: str | None
    nebius_base_url: str
    model: str
    fast_model: str
    image_model: str
    generate_images: bool
    tavily_api_key: str | None
    tavily_base_url: str
    tavily_keyless: bool
    sandbox_backend: str  # "local" | "nebius"
    gap_search: bool = True  # v2: Tavily + arXiv search for every MISSING PIECE
    use_archive: bool = True  # v2: Tesla Archive retrieval
    # v3: OpenMath-Nemotron math checker (self-hosted, OpenAI-compatible). Unset URL = checker skipped
    # (mock mode then shows canned derivations, clearly labelled).
    math_check: bool = True
    openmath_base_url: str | None = None
    openmath_model: str = OPENMATH_MODEL_DEFAULT
    openmath_api_key: str | None = None
    openmath_max_tokens: int = 12000
    openmath_temperature: float = 0.6
    openmath_timeout_s: float = 300.0
    math_check_tolerance: float = 0.02
    # v4: Patent Search Planner (+ Wildcard)
    patent_cache_dir: str = "cache/patents"  # every patent page read is cached here
    patent_fetch_interval_s: float = 1.0  # polite gap between patent-page requests
    planner_max_pages: int = 14  # patent pages read per run (seeds), plus up to 6 via the citation chain
    patents_bigquery: bool = False  # optional BigQuery path (needs google-cloud-bigquery + credentials)
    bq_project: str | None = None
    bq_max_gb: float = 20.0  # refuse any BigQuery query that would scan more than this
    wildcard: bool = True

    @property
    def openmath_configured(self) -> bool:
        return bool(self.openmath_base_url)

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"


def get_settings(force_mode: str | None = None) -> Settings:
    nebius_key = os.getenv("NEBIUS_API_KEY") or None
    tavily_key = os.getenv("TAVILY_API_KEY") or None
    mode = (force_mode or os.getenv("TESLA_LAB_MODE", "auto")).strip().lower()
    if mode not in {"mock", "live", "auto"}:
        raise ValueError(f"TESLA_LAB_MODE must be mock, live or auto, not {mode!r}")
    if mode == "auto":
        mode = "live" if (nebius_key and (tavily_key or _flag("TAVILY_KEYLESS"))) else "mock"
    return Settings(
        mode=mode,
        nebius_api_key=nebius_key,
        nebius_base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL_DEFAULT),
        model=os.getenv("NEMOTRON_MODEL", NEMOTRON_MODEL_DEFAULT),
        fast_model=os.getenv("NEMOTRON_FAST_MODEL", NEMOTRON_FAST_MODEL_DEFAULT),
        image_model=os.getenv("IMAGE_MODEL", IMAGE_MODEL_DEFAULT),
        generate_images=_flag("GENERATE_IMAGES"),
        tavily_api_key=tavily_key,
        tavily_base_url=os.getenv("TAVILY_BASE_URL", TAVILY_BASE_URL_DEFAULT),
        tavily_keyless=_flag("TAVILY_KEYLESS"),
        sandbox_backend=os.getenv("SANDBOX_BACKEND", "local").strip().lower(),
        gap_search=_flag("GAP_SEARCH", "1"),
        use_archive=_flag("USE_ARCHIVE", "1"),
        math_check=_flag("MATH_CHECK", "1"),
        openmath_base_url=(os.getenv("OPENMATH_BASE_URL") or "").strip() or None,
        openmath_model=(os.getenv("OPENMATH_MODEL") or "").strip() or OPENMATH_MODEL_DEFAULT,
        openmath_api_key=(os.getenv("OPENMATH_API_KEY") or "").strip() or None,
        openmath_max_tokens=int(os.getenv("OPENMATH_MAX_TOKENS", "12000")),
        openmath_temperature=float(os.getenv("OPENMATH_TEMPERATURE", "0.6")),
        openmath_timeout_s=float(os.getenv("OPENMATH_TIMEOUT_S", "300")),
        math_check_tolerance=float(os.getenv("MATH_CHECK_TOLERANCE", "0.02")),
        patent_cache_dir=os.getenv("PATENT_CACHE_DIR", "cache/patents"),
        patent_fetch_interval_s=float(os.getenv("PATENT_FETCH_INTERVAL_S", "1.0")),
        planner_max_pages=int(os.getenv("PLANNER_MAX_PAGES", "14")),
        patents_bigquery=_flag("PATENTS_BIGQUERY"),
        bq_project=(os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or None,
        bq_max_gb=float(os.getenv("PATENTS_BQ_MAX_GB", "20")),
        wildcard=_flag("WILDCARD", "1"),
    )
