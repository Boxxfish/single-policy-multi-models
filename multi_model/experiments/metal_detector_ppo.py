"""
Experiment where two different models handle the metal detector task.
"""
from argparse import ArgumentParser
from functools import reduce
from typing import *
import numpy as np

import torch
import torch.nn as nn
import wandb
from torch.distributions import Categorical
from tqdm import tqdm
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from multi_model.algorithms.ppo import train_ppo
from multi_model.algorithms.rollout_buffer import RolloutBuffer
from multi_model.conf import entity
from multi_model.envs import MetalDetectorEnv
from multi_model.utils import init_orthogonal

_: Any

# Hyperparameters
num_envs = 256  # Number of environments to step through at once during sampling.
train_steps = 128  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 1000  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.98  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
max_eval_steps = 500  # Number of eval runs to average over.
eval_steps = 8  # Max number of steps to take during each eval run.
entropy_coeff = 0.01 # Entropy coefficient.
v_lr = 0.01  # Learning rate of the value net.
p_lr = 0.001  # Learning rate of the policy net.
device = torch.device("cuda")  # Device to use during training.

# Handle args
parser = ArgumentParser()
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
setting = "baseline" if args.baseline else "two models"

wandb.init(
    project="multi_model",
    entity=entity,
    config={
        "experiment": f"metal_detector_ppo ({setting})",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "lambda": lambda_,
        "epsilon": epsilon,
        "max_eval_steps": max_eval_steps,
        "v_lr": v_lr,
        "p_lr": p_lr,
        "entropy_coeff": entropy_coeff,
    },
)


# The value network takes in an observation and returns a single value, the
# predicted return
class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        num_channels = obs_shape[0]
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(32, 1),
        )

    def forward(self, input: torch.Tensor):
        return self.net2(self.net(input).amax(-1).amax(-1))


# The policy network takes in an observation and returns the log probability of
# taking each action
class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        num_channels = obs_shape[0]
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(32, action_count),
            nn.LogSoftmax(1)
        )

    def forward(self, input: torch.Tensor):
        return self.net2(self.net(input).amax(-1).amax(-1))

grid_size = 5
env = SyncVectorEnv([lambda: MetalDetectorEnv(grid_size) for _ in range(num_envs)])
test_env = MetalDetectorEnv(grid_size)

# Initialize policy and value networks
obs_space = env.single_observation_space
act_space = env.single_action_space
assert isinstance(obs_space, gym.spaces.Box)
assert isinstance(act_space, gym.spaces.Discrete)
v_nets = [ValueNet(torch.Size(obs_space.shape)) for _ in range(2)]
p_nets = [PolicyNet(torch.Size(obs_space.shape), int(act_space.n)) for _ in range(2)]
v_opts = [torch.optim.Adam(v_net.parameters(), lr=v_lr) for v_net in v_nets]
p_opts = [torch.optim.Adam(p_net.parameters(), lr=p_lr) for p_net in p_nets]

# A rollout buffer stores experience collected during a sampling run
buffer = RolloutBuffer(
    torch.Size(obs_space.shape),
    torch.Size((1,)),
    torch.Size((int(act_space.n),)),
    torch.int,
    num_envs,
    train_steps,
)

obs = torch.Tensor(env.reset()[0])
done = False
for _ in tqdm(range(iterations), position=0):
    # Collect experience for a number of steps and store it in the buffer
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
            action_probs = p_net(obs)
            actions = Categorical(logits=action_probs).sample().numpy()
            obs_, rewards, dones, truncs, info = env.step(actions)
            buffer.insert_step(
                obs,
                torch.from_numpy(actions).unsqueeze(-1),
                action_probs,
                rewards.tolist(),
                dones.tolist(),
                truncs.tolist(),
            )
            obs = torch.from_numpy(obs_)
        buffer.insert_final_step(obs)

    # Train
    total_p_loss, total_v_loss = train_ppo(
        p_nets,
        v_nets,
        p_opts,
        v_opts,
        buffer,
        device,
        train_iters,
        train_batch_size,
        discount,
        lambda_,
        epsilon,
        entropy_coeff,
    )
    buffer.clear()

    # # Evaluate the network's performance after this training iteration.
    # eval_obs = torch.Tensor(test_env.reset()[0])
    # eval_done = False
    # with torch.no_grad():
    #     # Visualize
    #     reward_total = 0.0
    #     entropy_total = 0.0
    #     eval_obs = torch.Tensor(test_env.reset()[0])
    #     found_target_pct = 0
    #     for _ in range(eval_steps):
    #         avg_entropy = 0.0
    #         steps_taken = 0
    #         for _ in range(max_eval_steps):
    #             distr = Categorical(logits=p_net(eval_obs.unsqueeze(0)).squeeze())
    #             action = distr.sample().item()
    #             obs_, reward, eval_done, _, eval_info = test_env.step(action)
    #             eval_obs = torch.Tensor(obs_)
    #             steps_taken += 1
    #             reward_total += reward
    #             if eval_done:
    #                 found_target_pct += 1 if eval_info["found_target"] else 0
    #                 eval_obs = torch.Tensor(test_env.reset()[0])
    #                 break
    #             avg_entropy += distr.entropy()
    #         avg_entropy /= steps_taken
    #         entropy_total += avg_entropy

    wandb.log(
        {
            # "avg_eval_episode_reward": reward_total / eval_steps,
            # "avg_eval_entropy": entropy_total / eval_steps,
            # "found_target_pct": found_target_pct / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
        }
    )

class MultiSyncVector(gym.Wrapper):
    """
    A vectorized wrapper specifically for multiple tasks.
    Produces a list of states, along with a list of policy indices.
    """
    def __init__(self, env_list: list[Callable[[], gym.Env]]):
        self.envs = [env_fn() for env_fn in env_list]

    def step(self, action: int) -> tuple[list[np.ndarray], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        for env in self.envs:
            obs, reward, done, trunc, info = env.step()
    
    def reset(self, *args, **kwargs) -> tuple[list[Any], list[dict[str, Any]]]:
        all_obs = []
        all_infos = []
        for env in self.envs:
            obs, info = env.reset()
            all_obs.append(obs)
            all_infos.append(info)
        return (all_obs, all_infos)