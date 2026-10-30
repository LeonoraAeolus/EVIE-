import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("TESLA_LAB_MODE", "mock")
os.environ.setdefault("PYTHONUTF8", "1")
# v3: the test suite never calls a real OpenMath server, even if your .env sets one
# (python-dotenv does not override variables that are already set).
os.environ["OPENMATH_BASE_URL"] = ""
