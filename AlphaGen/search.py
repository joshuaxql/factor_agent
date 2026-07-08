from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .expressions import ActionCodec, Expr, ExpressionBuilder, ExpressionError
from .pool import MseAlphaPool


@dataclass
class Transition:
    state: list[int]
    mask: list[bool]
    action: int
    old_log_prob: Any


@dataclass
class Episode:
    expr: Expr | None
    reward: float
    transitions: list[Transition]
    length: int


class PolicyNet:
    def __init__(self, n_actions: int, hidden_size: int, num_layers: int, device: str, lr: float) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("AlphaGen generation requires PyTorch. Install torch for your CUDA/CPU environment.") from exc

        self.torch = torch
        self.device = torch.device("cuda:0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.n_actions = n_actions

        class Net(nn.Module):
            def __init__(self, n_actions: int, hidden_size: int, num_layers: int) -> None:
                super().__init__()
                self.embedding = nn.Embedding(n_actions + 1, hidden_size, padding_idx=0)
                self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
                self.head = nn.Linear(hidden_size, n_actions)

            def forward(self, state):
                emb = self.embedding(state)
                out, _ = self.lstm(emb)
                return self.head(out[:, -1])

        self.net = Net(n_actions, hidden_size, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def sample(self, state: list[int], mask: list[bool]):
        torch = self.torch
        state_tensor = torch.tensor([state], dtype=torch.long, device=self.device)
        logits = self.net(state_tensor)[0]
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(~mask_tensor, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, states, masks, actions):
        torch = self.torch
        states_tensor = torch.tensor(states, dtype=torch.long, device=self.device)
        masks_tensor = torch.tensor(masks, dtype=torch.bool, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        logits = self.net(states_tensor)
        logits = logits.masked_fill(~masks_tensor, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions_tensor), dist.entropy()

    def update(self, episodes: list[Episode], baseline: float, entropy_coef: float, ppo_clip: float, ppo_epochs: int) -> None:
        if not episodes:
            return
        torch = self.torch
        states = []
        masks = []
        actions = []
        old_log_probs = []
        advantages = []
        for episode in episodes:
            adv = float(episode.reward - baseline)
            for transition in episode.transitions:
                states.append(transition.state)
                masks.append(transition.mask)
                actions.append(transition.action)
                old_log_probs.append(transition.old_log_prob)
                advantages.append(adv)
        if not states:
            return
        old_log_probs_tensor = torch.stack(old_log_probs).detach().to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if advantages_tensor.std().item() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        for _ in range(ppo_epochs):
            log_probs, entropies = self.evaluate_actions(states, masks, actions)
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            unclipped = ratio * advantages_tensor
            clipped = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages_tensor
            loss = -torch.min(unclipped, clipped).mean() - entropy_coef * entropies.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class AlphaGenerator:
    def __init__(
        self,
        pool: MseAlphaPool,
        *,
        include_csrank: bool,
        max_expr_length: int,
        hidden_size: int,
        num_layers: int,
        device: str,
        lr: float,
        entropy_coef: float,
        ppo_clip: float,
        ppo_epochs: int,
        print_expr: bool = False,
    ) -> None:
        self.pool = pool
        self.codec = ActionCodec(include_csrank=include_csrank)
        self.max_expr_length = max_expr_length
        self.policy = PolicyNet(len(self.codec), hidden_size, num_layers, device, lr)
        self.entropy_coef = entropy_coef
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.print_expr = print_expr
        self.baseline = 0.0

    def run(self, steps: int, batch_size: int, out_dir: Path, log_every: int = 100) -> None:
        episodes: list[Episode] = []
        rows: list[dict[str, Any]] = []
        for step in range(1, steps + 1):
            episode = self._run_episode()
            episodes.append(episode)
            rows.append(
                {
                    "episode": step,
                    "expr": "" if episode.expr is None else str(episode.expr),
                    "reward": episode.reward,
                    "pool_size": self.pool.size,
                    "best_ic": self.pool.best_ic_ret,
                    "best_obj": self.pool.best_obj,
                    "length": episode.length,
                }
            )
            if len(episodes) >= batch_size:
                batch_reward = float(np.mean([e.reward for e in episodes]))
                self.baseline = 0.9 * self.baseline + 0.1 * batch_reward
                self.policy.update(episodes, self.baseline, self.entropy_coef, self.ppo_clip, self.ppo_epochs)
                episodes = []
            if step % log_every == 0 or step == 1:
                logger.info(f"[alphagen] episode={step} pool={self.pool.size} best_ic={self.pool.best_ic_ret:.6f}")
                self._write_outputs(out_dir, rows)
        if episodes:
            self.policy.update(episodes, self.baseline, self.entropy_coef, self.ppo_clip, self.ppo_epochs)
        self._write_outputs(out_dir, rows)

    def _run_episode(self) -> Episode:
        builder = ExpressionBuilder()
        state = [0] * self.max_expr_length
        transitions: list[Transition] = []
        for pos in range(self.max_expr_length):
            mask = builder.valid_action_mask(self.codec)
            if not any(mask):
                break
            state_before = list(state)
            action_idx, log_prob, entropy = self.policy.sample(state, mask)
            transitions.append(Transition(state_before, list(mask), action_idx, log_prob.detach()))
            action = self.codec.action(action_idx)
            if action.kind == "stop":
                return self._finish_episode(builder, transitions, pos + 1)
            try:
                builder.apply(action)
            except ExpressionError:
                return self._invalid_episode(transitions, pos + 1)
            state[pos] = action_idx + 1
        if builder.is_valid():
            return self._finish_episode(builder, transitions, self.max_expr_length)
        return self._invalid_episode(transitions, self.max_expr_length)

    def _finish_episode(self, builder: ExpressionBuilder, transitions: list[Transition], length: int) -> Episode:
        expr = builder.tree()
        if self.print_expr:
            print(expr)
        reward = self.pool.try_new_expr(expr)
        return Episode(expr, float(0.0 if np.isnan(reward) else reward), transitions, length)

    def _invalid_episode(self, transitions: list[Transition], length: int) -> Episode:
        return Episode(None, -1.0, transitions, length)

    def _write_outputs(self, out_dir: Path, rows: list[dict[str, Any]]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "episodes.csv", index=False)
        (out_dir / "pool.json").write_text(
            __import__("json").dumps(self.pool.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
