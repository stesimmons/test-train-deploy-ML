import io
import os
import shutil
import subprocess
import webbrowser
import zipfile

from pathlib import Path

import requests


OWNER = "stesimmons"
REPO = "test-train-deploy-ML"

TOKEN = os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN environment variable not found."
    )

PROJECT_ROOT = Path(__file__).parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

if MLRUNS_DIR.exists():
    shutil.rmtree(MLRUNS_DIR)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
}

print("Getting workflow runs...")

runs_url = (
    f"https://api.github.com/repos/"
    f"{OWNER}/{REPO}/actions/runs"
)

runs = requests.get(
    runs_url,
    headers=headers,
    timeout=30
).json()

successful_run = None

for run in runs["workflow_runs"]:

    if run["conclusion"] == "success":

        successful_run = run
        break

if successful_run is None:
    raise RuntimeError(
        "No successful workflow runs found."
    )

run_id = successful_run["id"]

print(f"Using workflow run: {run_id}")

artifacts_url = (
    f"https://api.github.com/repos/"
    f"{OWNER}/{REPO}/actions/runs/"
    f"{run_id}/artifacts"
)

artifacts = requests.get(
    artifacts_url,
    headers=headers,
    timeout=30
).json()

artifact = None

for item in artifacts["artifacts"]:

    if item["name"].startswith(
        "mlflow-runs"
    ):
        artifact = item
        break

if artifact is None:
    raise RuntimeError(
        "No mlflow-runs artifact found."
    )

print(
    f"Downloading artifact: "
    f"{artifact['name']}"
)

zip_response = requests.get(
    artifact["archive_download_url"],
    headers=headers,
    timeout=60
)

with zipfile.ZipFile(
    io.BytesIO(zip_response.content)
) as z:

    z.extractall(PROJECT_ROOT)

print("Artifact extracted.")

print("Starting MLflow UI...")

subprocess.Popen(
    ["mlflow", "ui"]
)

webbrowser.open(
    "http://127.0.0.1:5000"
)

print(
    "\nMLflow UI started."
    "\nOpen: http://127.0.0.1:5000"
)