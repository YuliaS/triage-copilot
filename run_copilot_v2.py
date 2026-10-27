import json, warnings
warnings.filterwarnings("ignore")

nb = json.load(open("triage-copilot-v2.ipynb"))
ns = {}
for c in nb["cells"]:
    if c["cell_type"] == "code":
        exec("".join(c["source"]).replace("# demo.launch", "pass  # demo.launch"), ns)

print("serving triage-copilot-v2 (loads 5 saved models) on 7866")
ns["demo"].launch(server_name="127.0.0.1", server_port=7866, share=False, inbrowser=False)
