import os
import glob
import csv

import matplotlib.pyplot as plt


def find_latest_metrics_file(logs_dir="logs"):
    """Return path to the newest metrics_*.csv file in logs_dir, or None."""
    pattern = os.path.join(logs_dir, "metrics_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_metrics(path):
    """Load metrics CSV into lists."""
    steps = []
    victims_found = []
    coverage = []
    elapsed = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            victims_found.append(int(row["victims_found"]))
            coverage.append(float(row["coverage_percent"]))
            elapsed.append(float(row["elapsed_seconds"]))

    return steps, victims_found, coverage, elapsed


def plot_metrics(path):
    steps, victims_found, coverage, elapsed = load_metrics(path)

    plt.figure(figsize=(10, 5))

    # Victims found vs. step
    plt.subplot(1, 2, 1)
    plt.plot(steps, victims_found, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Victims found")
    plt.title("Victims found vs. Step")
    plt.grid(True)

    # Coverage vs. step
    plt.subplot(1, 2, 2)
    plt.plot(steps, coverage, marker="o", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Coverage (%)")
    plt.title("Coverage vs. Step")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    path = find_latest_metrics_file()
    if path is None:
        print("No metrics_*.csv files found in 'logs' directory.")
        print("Run: python main.py  first to generate metrics.")
        return

    print("Using metrics file:", path)
    plot_metrics(path)


if __name__ == "__main__":
    main()
