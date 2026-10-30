"""Check your Tavily key with one search.   python scripts/check_tavily.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tesla_lab.config import get_settings  # noqa: E402
from tesla_lab.research import TavilyClient  # noqa: E402

data = TavilyClient(get_settings("live")).search("Wardenclyffe Tower wireless power", max_results=3)
for r in data.get("results", []):
    print(f"{r.get('score', 0):.2f}  {r['title'][:70]}  {r['url']}")
