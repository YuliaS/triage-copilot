"""Headless launcher: executes app/app.ipynb and serves it."""
import json, os, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, "app.ipynb")))
ns = {}
for c in nb["cells"]:
    if c["cell_type"] == "code":
        exec("".join(c["source"]).replace("# demo.launch", "pass  # demo.launch"), ns)
print("serving Northstar Triage Copilot on 7866")
ns["demo"].launch(server_name="127.0.0.1", server_port=7866, share=False, inbrowser=False)
