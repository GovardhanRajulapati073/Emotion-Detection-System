from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app import app

app.static_folder = str(ROOT / "static")
app.template_folder = str(ROOT / "templates")

application = app
