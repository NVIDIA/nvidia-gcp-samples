"""Run the API smoke test from the pinned upstream checkout."""
import os
import runpy
from pathlib import Path

source = Path(os.environ.get("RFD4_SOURCE", "upstream")).resolve()
os.environ["RFD4_SOURCE"] = str(source)
runpy.run_path(str(source / "deploy/api/smoke_test.py"), run_name="__main__")
