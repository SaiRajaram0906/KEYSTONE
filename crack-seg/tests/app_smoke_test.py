"""Does the front end start and survive the example comparison?   python tests/app_smoke_test.py"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
assert not at.exception, [e.value for e in at.exception]
next(b for b in at.button if "example" in b.label).click()
at.run()
assert not at.exception, [e.value for e in at.exception]
html = " ".join(m.value for m in at.markdown)
assert "The crack got wider" in html, "example comparison should report widening"
print("app smoke test passed")
sys.exit(0)
