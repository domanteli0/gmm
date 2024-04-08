install-deps:
    python3 -m pip install -r requirements.txt

# Meant to be copy pasted into the command line
source-fish:
    source .venv/bin/activate.fish

start-lab2:
    bokeh serve --show lab2_ui.py
