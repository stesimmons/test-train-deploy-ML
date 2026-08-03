import json
import sys

MIN_ACCURACY = 80.0

with open(
    "metrics.json",
    "r"
) as f:

    metrics = json.load(f)

accuracy = metrics["best_accuracy"]

if accuracy < MIN_ACCURACY:

    print(
        f"Accuracy "
        f"{accuracy}% "
        f"below minimum "
        f"{MIN_ACCURACY}%"
    )

    sys.exit(1)

print(
    f"Accuracy check passed "
    f"({accuracy}%)"
)