# Validates that the selected production model meets the minimum accuracy
# threshold required by the CI/CD pipeline. If the best model's accuracy
# falls below the configured value, the workflow fails and deployment is
# blocked to prevent promoting an underperforming model.

import json
import sys

MIN_ACCURACY = 80.0

with open(
    "metrics.json",
    "r"
) as f:

    metrics = json.load(f)

accuracy = metrics["best_accuracy"]

architecture = metrics.get(
    "best_architecture",
    "Unknown"
)

version = metrics.get(
    "version",
    "Unknown"
)

stage = metrics.get(
    "stage",
    "Unknown"
)

if accuracy < MIN_ACCURACY:

    print(
        f"Accuracy {accuracy}% "
        f"below minimum "
        f"{MIN_ACCURACY}%"
    )

    sys.exit(1)

print(
    f"Accuracy check passed "
    f"({accuracy}%)"
)

print(
    f"Architecture: "
    f"{architecture}"
)

print(
    f"Version: "
    f"{version}"
)

print(
    f"Stage: "
    f"{stage}"
)