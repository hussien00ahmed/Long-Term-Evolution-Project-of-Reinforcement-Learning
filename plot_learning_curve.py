import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--eval-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    train_rows = read_csv_rows(log_dir / "train.log")
    eval_rows = read_csv_rows(log_dir / "eval.log")

    train_steps = [int(row["step"]) for row in train_rows]
    train_rewards = [float(row["episode_reward"]) for row in train_rows]

    eval_steps = [int(row["step"]) for row in eval_rows]
    eval_rewards = [float(row["episode_reward"]) for row in eval_rows]

    final_eval_label = None
    if args.eval_json:
        eval_data = json.loads(Path(args.eval_json).read_text())
        eval_steps.append(train_steps[-1])
        eval_rewards.append(float(eval_data["trained_policy"]["reward_mean"]))
        final_eval_label = (
            f"Post-train eval ({eval_data['trained_policy']['episodes']} ep): "
            f"{eval_data['trained_policy']['reward_mean']:.2f}"
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    ax.plot(train_steps, train_rewards, marker="o", linewidth=2, color="#1f77b4", label="Train reward")
    ax.plot(eval_steps, eval_rewards, marker="s", linewidth=2, color="#d62728", label="Eval reward")

    if final_eval_label is not None:
        ax.annotate(
            final_eval_label,
            xy=(eval_steps[-1], eval_rewards[-1]),
            xytext=(12, -18),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d62728", "alpha": 0.9},
        )

    ax.set_title("DrQv2 Learning Curve on cartpole_balance (pixels)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
