# Monitors GitHub Actions for new successful workflow runs and downloads
# the latest MLflow artifact to the local machine. This keeps the local
# MLflow UI synchronized with the CI/CD pipeline so experiment results,
# model versions, metrics, and production promotions can be reviewed
# outside of GitHub.
# This script needs to be running before any models are pushed to 
# GitHub for accurate tracking.

import io
import json
import os
import shutil
import time
import zipfile

from pathlib import Path

import requests


OWNER = "stesimmons"
REPO = "test-train-deploy-ML"

CHECK_INTERVAL = 300  # 5 minutes

TOKEN = os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN environment variable not found."
    )

PROJECT_ROOT = Path(__file__).parent

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

STATE_FILE = (
    PROJECT_ROOT
    / ".mlflow_sync_state.json"
)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
}


def load_last_run():

    if not STATE_FILE.exists():
        return None

    with open(
        STATE_FILE,
        "r"
    ) as f:

        data = json.load(f)

    return data.get(
        "last_run_id"
    )


def save_last_run(run_id):

    with open(
        STATE_FILE,
        "w"
    ) as f:

        json.dump(
            {
                "last_run_id": run_id
            },
            f
        )


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


def download_mlflow_artifact(run_id):

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
            "No mlflow-runs artifact found."
        )

        return False

    print(
        f"Downloading: "
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

    if MLRUNS_DIR.exists():

        shutil.rmtree(
            MLRUNS_DIR
        )

    MLRUNS_DIR.mkdir(
        exist_ok=True
    )

    with zipfile.ZipFile(
        io.BytesIO(
            zip_response.content
        )
    ) as z:

        z.extractall(
            MLRUNS_DIR
        )

    print(
        f"MLflow runs extracted to: "
        f"{MLRUNS_DIR}"
    )

    return True


def main():

    print(
        "MLflow watcher started."
    )

    last_run_id = (
        load_last_run()
    )

    while True:

        try:

            run = (
                get_latest_successful_run()
            )

            if run:

                current_run_id = (
                    run["id"]
                )

                if (
                    last_run_id is None
                    or current_run_id
                    != last_run_id
                ):

                    print(
                        "New workflow detected: "
                        f"{current_run_id}"
                    )

                    success = (
                        download_mlflow_artifact(
                            current_run_id
                        )
                    )

                    if success:

                        save_last_run(
                            current_run_id
                        )

                        last_run_id = (
                            current_run_id
                        )

                else:

                    print(
                        "No new workflow."
                    )

            time.sleep(
                CHECK_INTERVAL
            )

        except Exception as e:

            print(
                f"Watcher error: {e}"
            )

            time.sleep(
                CHECK_INTERVAL
            )


if __name__ == "__main__":
    main()