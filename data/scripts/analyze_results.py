import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results/charts")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

scores = pd.read_csv(DATA_DIR / "scores.csv")
prompts = pd.read_csv(DATA_DIR / "prompts.csv")

# Remove rows that are not scored yet
scored = scores.dropna(subset=["accuracy", "clarity", "educational_value", "hallucination"])

if scored.empty:
    print("No scores found yet. Fill data/scores.csv first.")
else:
    merged = scored.merge(prompts, left_on="prompt_id", right_on="id")

    metrics = ["accuracy", "clarity", "educational_value", "hallucination"]

    print("\nAverage scores by model:")
    print(merged.groupby("model")[metrics].mean().round(2))

    print("\nAverage scores by model and task type:")
    print(merged.groupby(["model", "task_type"])[metrics].mean().round(2))

    print("\nAverage scores by model and level:")
    print(merged.groupby(["model", "level"])[metrics].mean().round(2))

    for metric in metrics:
        ax = merged.groupby("model")[metric].mean().plot(kind="bar", title=f"Average {metric} by model")
        ax.set_ylabel(metric)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{metric}_by_model.png")
        plt.clf()

    print("\nCharts saved in results/charts/")
