"""统一评估器。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from robocore.env.base import BaseEnv
from robocore.env.vec_env import VecEnv
from robocore.policy.base import BasePolicy

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果。"""

    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_episode_length: float = 0.0
    num_episodes: int = 0
    per_episode: list[dict[str, Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"EvalResult(success_rate={self.success_rate:.3f}, "
            f"mean_reward={self.mean_reward:.2f}, "
            f"episodes={self.num_episodes})"
        )


class Evaluator:
    """统一评估器。支持单环境和向量化环境评估。"""

    def __init__(
        self,
        env: BaseEnv | VecEnv,
        num_episodes: int = 50,
        max_steps: int = 400,
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    @torch.no_grad()
    def run(self, policy: BasePolicy) -> EvalResult:
        """运行评估。"""
        policy.eval()
        if isinstance(self.env, VecEnv):
            episodes_data = self._run_vec(policy)
        else:
            episodes_data = self._run_single(policy)

        successes = [ep.get("success", False) for ep in episodes_data]
        rewards = [ep.get("total_reward", 0.0) for ep in episodes_data]
        lengths = [ep.get("length", 0) for ep in episodes_data]

        return EvalResult(
            success_rate=float(np.mean(successes)) if successes else 0.0,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_episode_length=float(np.mean(lengths)) if lengths else 0.0,
            num_episodes=len(episodes_data),
            per_episode=episodes_data,
        )

    def _run_single(self, policy: BasePolicy) -> list[dict[str, Any]]:
        """单环境串行评估。"""
        results = []
        for ep_idx in range(self.num_episodes):
            obs = self.env.reset(seed=ep_idx)
            total_reward = 0.0
            done = False
            step = 0
            result_info: dict[str, Any] = {}

            while not done and step < self.max_steps:
                obs_batch = {
                    k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0)
                    if isinstance(v, np.ndarray)
                    else v.unsqueeze(0).unsqueeze(0)
                    for k, v in obs.items()
                    if isinstance(v, (np.ndarray, torch.Tensor))
                }
                output = policy.predict(obs_batch)
                action = policy.get_action(output)
                action_np = action[0, 0].cpu().numpy()

                step_result = self.env.step(action_np)
                obs = step_result.obs
                total_reward += float(step_result.reward)
                done = bool(step_result.done) or bool(step_result.truncated)
                result_info = step_result.info
                step += 1

            results.append({
                "episode": ep_idx,
                "success": result_info.get("success", False),
                "total_reward": total_reward,
                "length": step,
            })

            if (ep_idx + 1) % 10 == 0:
                sr = np.mean([r["success"] for r in results])
                logger.info(f"Progress [{ep_idx+1}/{self.num_episodes}] success_rate={sr:.3f}")

        return results

    def _run_vec(self, policy: BasePolicy) -> list[dict[str, Any]]:
        """向量化环境并行评估。"""
        assert isinstance(self.env, VecEnv)
        results = []
        completed = 0

        obs = self.env.reset(seed=0)
        episode_rewards = np.zeros(self.env.num_envs)
        episode_lengths = np.zeros(self.env.num_envs, dtype=int)

        while completed < self.num_episodes:
            obs_batch = {
                k: torch.from_numpy(v).unsqueeze(1)
                if isinstance(v, np.ndarray)
                else v.unsqueeze(1)
                for k, v in obs.items()
                if isinstance(v, (np.ndarray, torch.Tensor))
            }
            output = policy.predict(obs_batch)
            action = policy.get_action(output)
            action_np = action[:, 0].cpu().numpy()

            obs, rewards, dones, truncateds, infos = self.env.step(action_np)
            episode_rewards += rewards
            episode_lengths += 1

            for i in range(self.env.num_envs):
                if dones[i] or truncateds[i] or episode_lengths[i] >= self.max_steps:
                    results.append({
                        "episode": completed,
                        "success": infos[i].get("success", False),
                        "total_reward": float(episode_rewards[i]),
                        "length": int(episode_lengths[i]),
                    })
                    completed += 1
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                    if completed >= self.num_episodes:
                        break

        return results[:self.num_episodes]
