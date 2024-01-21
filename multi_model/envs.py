import copy
from typing import *
from typing import Any
import gymnasium as gym
import numpy as np
import random


class MetalDetectorEnv(gym.Env):
    """
    An NxN gridworld environment that switches between a guided pathfinding task to a pure pathfinding task.

    In the first task, on each step, the agent is given the current distance to the target cell. These measurements are
    given for each cell the agent has stepped on within the episode. The agent must use these measurements to locate the
    target as quickly as possible.

    In the second task, the measurements are removed. However, if the agent had reached the correct target cell in the
    previous task, this is now indicated on the map, making this a pure pathfinding challenge.

    The difficulty of this environment lies in the fact that these two tasks must be more or less solved at the same
    time to get the reward. While the first task is not solved, the second task will not be able to locate the correct
    cell. And while the second task is not solved, the first task will not be able to identify how to relate returns to
    its observations.

    ### Actions
    0. Left (-x)
    1. Right (+x)
    2. Up (+y)
    3. Down (-y)
    """

    def __init__(self, size: int):
        self.target_pos = (0, 0)
        self.agent_pos = [0, 0]
        self.orig_pos = [0, 0]
        self.cell_dists = [[0 for _ in range(size)] for _ in range(size)]
        self.cell_dists_visible = [[0 for _ in range(size)] for _ in range(size)]
        self.task_idx = 0
        self.size = size
        self.found_target = False
        self.time_left = 0
        self.max_time = (self.size - 1) * 2
        self.observation_space = gym.spaces.Box(0.0, 1.0, [2, size, size])
        self.action_space = gym.spaces.Discrete(4)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Move the agent
        if action == 0:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        if action == 1:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.size - 1)
        if action == 2:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.size - 1)
        if action == 3:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)

        rew = 0
        done = False

        if self.task_idx == 0:
            # Reveal the distance for the current cell
            self.cell_dists_visible[self.agent_pos[1]][
                self.agent_pos[0]
            ] = self.cell_dists[self.agent_pos[1]][self.agent_pos[0]]

            # If the timer has run out, finish this task
            self.time_left -= 1
            if self.time_left == 0:
                self.time_left = self.max_time
                self.task_idx = 1
                self.agent_pos = copy.deepcopy(self.orig_pos)

            # If the agent is touching the target, finish this task
            if tuple(self.agent_pos) == self.target_pos:
                self.found_target = True
                self.task_idx = 1
                self.agent_pos = copy.deepcopy(self.orig_pos)

        elif self.task_idx == 1:
            # If the timer has run out, finish this task
            self.time_left -= 1
            if self.time_left == 0:
                rew = -1
                done = True

            # If the agent is touching the target, finish this task
            if tuple(self.agent_pos) == self.target_pos:
                rew = 1
                done = True

        obs, info = self.gen_observation_and_info()
        return obs, rew, done, False, info

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.target_pos = (
            random.randrange(0, self.size),
            random.randrange(0, self.size),
        )
        self.agent_pos = [
            random.randrange(0, self.size),
            random.randrange(0, self.size),
        ]
        while tuple(self.agent_pos) == self.target_pos:
            self.agent_pos = [
                random.randrange(0, self.size),
                random.randrange(0, self.size),
            ]
        self.orig_pos = copy.deepcopy(self.agent_pos)
        self.cell_dists = [
            [
                abs(self.target_pos[0] - x) + abs(self.target_pos[1] - y)
                for x in range(self.size)
            ]
            for y in range(self.size)
        ]
        self.cell_dists_visible = [
            [-((self.size - 1) ** 2) for _ in range(self.size)]
            for _ in range(self.size)
        ]  # Ends up as -1 once division is taken into account
        self.cell_dists_visible[self.agent_pos[1]][self.agent_pos[0]] = self.cell_dists[
            self.agent_pos[1]
        ][self.agent_pos[0]]
        self.found_target = False
        self.time_left = self.max_time
        self.task_idx = 0
        return self.gen_observation_and_info()

    def gen_observation_and_info(self) -> tuple[np.ndarray, dict[str, Any]]:
        pos_channel = np.zeros([self.size, self.size])
        pos_channel[self.agent_pos[1], self.agent_pos[0]] = 1
        if self.task_idx == 0:
            dists_channel = np.array(self.cell_dists_visible) / (self.size - 1) ** 2
            obs = np.stack([pos_channel, dists_channel], 0)
        else:
            target_channel = np.zeros([self.size, self.size])
            if self.found_target:
                target_channel[self.target_pos[1], self.target_pos[0]] = 1
            obs = np.stack([pos_channel, target_channel], 0)
        return obs, {"task_idx": self.task_idx}
