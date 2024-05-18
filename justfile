install-deps:
    python3 -m pip install -r requirements.txt

# Meant to be copy pasted into the command line
source-fish:
    source .venv/bin/activate.fish

start-lab2:
    bokeh serve --show lab2_ui.py

serve-lab3:
    python lab3_service.py

prepare-lab3:
    python lab3/prepare_data.py

list-notebooks:
    venv/bin/jupyter notebook list
