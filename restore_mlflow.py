# Restores the latest MLflow tracking store from GitHub Actions artifacts
# before training. This preserves experiment history across workflow runs,
# allowing MLflow to accumulate and compare model versions, architectures,
# metrics, and production promotions over time instead of starting with an
# empty mlruns/ directory on every CI execution.

import io
import os
import shutil
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

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
}


def get_latest_successful_run():

    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/actions/runs"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    runs = response.json()[
        "workflow_runs"
    ]

    for run in runs:

        if (
            run["status"]
            == "completed"
            and
            run["conclusion"]
            == "success"
        ):
            return run

    return None


def restore_mlflow_store():

    run = (
        get_latest_successful_run()
    )

    if run is None:

        print(
            "No previous successful workflow found."
        )

        return

    run_id = run["id"]

    print(
        f"Using workflow: {run_id}"
    )

    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/actions/runs/"
        f"{run_id}/artifacts"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    artifacts = response.json()[
        "artifacts"
    ]

    artifact = None

    for item in artifacts:

        if item["name"].startswith(
            "mlflow-runs"
        ):
            artifact = item
            break

    if artifact is None:

        print(
            "No MLflow artifact found."
        )

        return

    print(
        f"Downloading "
        f"{artifact['name']}"
    )

    zip_response = requests.get(
        artifact[
            "archive_download_url"
        ],
        headers=HEADERS,
        timeout=60
    )

    zip_response.raise_for_status()

    temp_dir = (
        PROJECT_ROOT
        / "temp_mlruns"
    )

    if temp_dir.exists():

        shutil.rmtree(
            temp_dir
        )

    temp_dir.mkdir()

    with zipfile.ZipFile(
        io.BytesIO(
            zip_response.content
        )
    ) as z:

        z.extractall(
            temp_dir
        )

    if MLRUNS_DIR.exists():

        shutil.rmtree(
            MLRUNS_DIR
        )

    shutil.move(
        str(temp_dir),
        str(MLRUNS_DIR)
    )

    print(
        f"Restored MLflow store "
        f"to {MLRUNS_DIR}"
    )


if __name__ == "__main__":
    restore_mlflow_store()