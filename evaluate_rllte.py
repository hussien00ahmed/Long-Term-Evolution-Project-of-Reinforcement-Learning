import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch as th

from rllte.common import utils
from rllte.env import make_dmc_env
from rllte.xploit.storage import nstep_replay_storage as nstep_replay_storage_module


def _safe_worker_init_fn(worker_id: int) -> None:
    seed = int(np.random.get_state()[1][0]) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)


def make_env(device: str):
    return make_dmc_env(
        env_id="cartpole_balance",
        device=device,
        visualize_reward=False,
        from_pixels=True,
        asynchronous=sys.platform != "win32",
    )


def summarize(rewards: list[float], lengths: list[int]) -> dict[str, object]:
    return {
        "episodes": len(rewards),
        "reward_mean": mean(rewards),
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "length_mean": mean(lengths),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
    }


def run_policy(env, action_fn, episodes: int, seed: int) -> dict[str, object]:
    obs, infos = env.reset(seed=seed)
    rewards: list[float] = []
    lengths: list[int] = []

    while len(rewards) < episodes:
        actions = action_fn(obs)
        obs, rews, terms, truncs, infos = env.step(actions)
        if "episode" in infos:
            eps_r, eps_l = utils.get_episode_statistics(infos)
            rewards.extend(float(x) for x in eps_r)
            lengths.extend(int(x) for x in eps_l)

    return summarize(rewards[:episodes], lengths[:episodes])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    nstep_replay_storage_module.worker_init_fn = _safe_worker_init_fn

    model_path = Path(args.model_path).resolve()
    model = th.load(model_path, map_location=args.device, weights_only=False)
    model.eval()

    eval_env = make_env(args.device)
    random_env = make_env(args.device)

    def trained_action(obs):
        with th.no_grad():
            return model(obs)

    def random_action(obs):
        low = random_env.action_space.low
        high = random_env.action_space.high
        shape = (random_env.num_envs,) + random_env.action_space.shape
        return th.as_tensor(np.random.uniform(low, high, size=shape), dtype=th.float32)

    result = {
        "model_path": str(model_path),
        "task": "cartpole_balance from pixels",
        "trained_policy": run_policy(eval_env, trained_action, args.episodes, args.seed),
        "random_policy_baseline": run_policy(random_env, random_action, args.episodes, args.seed),
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
