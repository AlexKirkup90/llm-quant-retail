import streamlit as st
from pathlib import Path
import sys

# --- Make project root importable ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controller import Controller
from src.simple_controller import SimpleController
from src.simple_view import SimpleView

def main():
    if hasattr(st, "set_page_config"):
        st.set_page_config(page_title="LLM-Codex Quant", layout="wide")

    # Instantiate the original controller
    controller = Controller()

    # Instantiate the simple controller and view
    simple_controller = SimpleController(controller)
    simple_view = SimpleView(simple_controller)

    # Render the simplified UI
    simple_view.render()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
