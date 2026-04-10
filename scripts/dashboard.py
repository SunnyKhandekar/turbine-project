from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "streamlit.exe"), "run", str(ROOT / "src" / "turbine_project" / "dashboard_entry.py")],
        check=True,
    )
