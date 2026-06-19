import random
import sys

import numpy as np

from rllte.agent import DrQv2
from rllte.env import make_dmc_env
from rllte.xploit.storage import nstep_replay_storage as nstep_replay_storage_module


def _safe_worker_init_fn(worker_id: int) -> None:
    seed = int(np.random.get_state()[1][0]) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    device = "cpu"
    use_async_env = sys.platform != "win32"
    nstep_replay_storage_module.worker_init_fn = _safe_worker_init_fn

    env = make_dmc_env(
        env_id="cartpole_balance",
        device=device,
        visualize_reward=False,
        from_pixels=True,
        asynchronous=use_async_env,
    )
    eval_env = make_dmc_env(
        env_id="cartpole_balance",
        device=device,
        visualize_reward=False,
        from_pixels=True,
        asynchronous=use_async_env,
    )

    agent = DrQv2(
        env=env,
        eval_env=eval_env,
        device=device,
        tag="drqv2_dmc_pixel",
    )
    agent.storage.num_workers = 1
    agent.storage.dataset._num_workers = 1
    agent.storage.pin_memory = False
    agent.storage.reset()

    agent.train(num_train_steps=5000, log_interval=1000)


if __name__ == "__main__":
    main()
