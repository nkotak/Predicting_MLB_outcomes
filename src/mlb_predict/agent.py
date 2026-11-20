"""A betting agent trained with Group Relative Policy Optimization (GRPO).

What the agent does
-------------------
For each game it sees the model's win chance and the market price, and picks
one action: pass, or back the home or away side with a small, medium or large
paper stake. It is never told a rule such as "bet when the edge is above 3%".
It learns which situations were worth betting from how past bets turned out.

How it learns (GRPO)
--------------------
GRPO is a trial-and-reward method. For every game in the training data the
agent tries a *group* of actions drawn from its current policy. Each action
gets a reward: how much a bankroll would have grown or shrunk. An action is
then judged *relative to the rest of its group* for that same game: better
than the group average is encouraged, worse is discouraged. No separate
"critic" model is needed, which is the defining idea of GRPO.

Two safeguards from the method keep learning steady:

* clipping: one update cannot change the chance of an action by more than a
  set ratio, so a lucky streak cannot swing the policy
* a KL penalty: the new policy is kept close to the reference (previous) one

One deliberate difference from the original recipe
---------------------------------------------------
The original GRPO divides each group's rewards by that group's own spread.
Here the spread of rewards inside one game depends on who won, and dividing
by it would quietly reward backing favourites whatever the price. So rewards
are centred per game (the group-relative part) but scaled by the spread of
the whole batch. Later GRPO work makes the same change for the same reason.

The reward is the log of bankroll growth, which is what the Kelly rule
maximizes, so the agent is paid for growing the bankroll and punished more
for big losses than it is rewarded for equal-sized wins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .betting import edge, fair_probabilities
from .model import logit

STAKES: tuple[float, ...] = (0.005, 0.01, 0.02)  # share of bankroll: small, medium, large
ACTIONS: tuple[tuple[str, float], ...] = (
    (("pass", 0.0),) + tuple(("home", s) for s in STAKES) + tuple(("away", s) for s in STAKES)
)
N_ACTIONS = len(ACTIONS)
STATE_NAMES = ("model_minus_market", "abs_model_minus_market", "market_logit", "abs_market_logit",
               "edge_home", "edge_away", "best_edge")
REWARD_SCALE = 100.0  # rewards are in percent of bankroll


def state_matrix(p_model, home_dec, away_dec) -> np.ndarray:
    """What the agent sees for each game. Uses nothing that is unknown before first pitch."""
    p_model = np.asarray(p_model, dtype=float)
    home_dec, away_dec = np.asarray(home_dec, dtype=float), np.asarray(away_dec, dtype=float)
    fair_home, _ = fair_probabilities(home_dec, away_dec)
    z_model, z_market = logit(p_model), logit(fair_home)
    edge_home, edge_away = edge(p_model, home_dec), edge(1.0 - p_model, away_dec)
    return np.column_stack([
        z_model - z_market, np.abs(z_model - z_market), z_market, np.abs(z_market),
        edge_home, edge_away, np.maximum(edge_home, edge_away),
    ])


def reward_table(home_dec, away_dec, home_win) -> np.ndarray:
    """Reward of every action for every game: percent log growth of the bankroll."""
    home_dec, away_dec = np.asarray(home_dec, dtype=float), np.asarray(away_dec, dtype=float)
    home_win = np.asarray(home_win, dtype=float)
    table = np.zeros((len(home_dec), N_ACTIONS))
    for index, (side, stake) in enumerate(ACTIONS):
        if side == "pass":
            continue
        price = home_dec if side == "home" else away_dec
        won = home_win == (1.0 if side == "home" else 0.0)
        table[:, index] = REWARD_SCALE * np.where(won, np.log1p(stake * (price - 1.0)), np.log1p(-stake))
    return table


@dataclass
class GRPOConfig:
    hidden: int = 12  # size of the policy's one hidden layer
    group_size: int = 16  # actions tried per game in each round
    clip: float = 0.2  # largest allowed change in an action's chance per update (as a ratio)
    kl_weight: float = 0.05  # strength of the pull toward the reference policy
    entropy_weight: float = 0.0
    learning_rate: float = 0.01
    iterations: int = 80  # rounds of "try actions, then update"
    epochs: int = 3  # passes over each round's samples
    batch_size: int = 4096
    weight_decay: float = 1e-4
    patience: int = 15  # stop after this many rounds without a better validation result
    seed: int = 3


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


class _Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key in params:
            self.m[key] = 0.9 * self.m[key] + 0.1 * grads[key]
            self.v[key] = 0.999 * self.v[key] + 0.001 * grads[key] ** 2
            m_hat = self.m[key] / (1 - 0.9 ** self.t)
            v_hat = self.v[key] / (1 - 0.999 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def policy_log_probs(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log chance of each action for each game, and the hidden layer values."""
    hidden = np.tanh(x @ params["w1"] + params["b1"])
    return _log_softmax(hidden @ params["w2"] + params["b2"]), hidden


def grpo_loss_and_grads(params, x, actions, advantages, log_old, config: GRPOConfig):
    """The GRPO objective (as a loss to minimize) and its exact gradients.

    ``actions`` and ``advantages`` have one row per game and one column per
    group member. ``log_old`` holds the reference policy's log chances.
    """
    n, g = actions.shape
    log_new, hidden = policy_log_probs(params, x)
    probs = np.exp(log_new)
    ratio = np.exp(np.take_along_axis(log_new, actions, 1) - np.take_along_axis(log_old, actions, 1))
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - config.clip, 1.0 + config.clip) * advantages
    surrogate = np.minimum(unclipped, clipped)
    active = unclipped <= clipped  # where the unclipped term is the one in force

    log_ratio_all = log_new - log_old
    kl = (probs * log_ratio_all).sum(axis=1)
    entropy = -(probs * log_new).sum(axis=1)
    decay = 0.5 * config.weight_decay * ((params["w1"] ** 2).sum() + (params["w2"] ** 2).sum())
    loss = -surrogate.mean() + config.kl_weight * kl.mean() - config.entropy_weight * entropy.mean() + decay

    coef = active * advantages * ratio / (n * g)
    d_surrogate = -probs * coef.sum(axis=1, keepdims=True)
    np.add.at(d_surrogate, (np.repeat(np.arange(n), g), actions.ravel()), coef.ravel())
    d_kl = probs * (log_ratio_all - kl[:, None]) / n
    d_entropy = -probs * (log_new + entropy[:, None]) / n
    dz = -d_surrogate + config.kl_weight * d_kl - config.entropy_weight * d_entropy

    d_hidden = (dz @ params["w2"].T) * (1.0 - hidden ** 2)
    grads = {
        "w2": hidden.T @ dz + config.weight_decay * params["w2"],
        "b2": dz.sum(axis=0),
        "w1": x.T @ d_hidden + config.weight_decay * params["w1"],
        "b1": d_hidden.sum(axis=0),
    }
    return float(loss), grads


def group_relative_advantages(rewards: np.ndarray) -> np.ndarray:
    """Each sampled action's reward relative to its own group, on a common scale."""
    centred = rewards - rewards.mean(axis=1, keepdims=True)
    return centred / (centred.std() + 1e-8)


def sample_actions(probs: np.ndarray, group_size: int, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random((len(probs), group_size))
    return np.minimum((draws[:, :, None] > cumulative[:, None, :]).sum(axis=2), probs.shape[1] - 1)


@dataclass
class GRPOAgent:
    """A small policy network trained with GRPO to make paper-bet decisions."""

    config: GRPOConfig = field(default_factory=GRPOConfig)
    params: dict | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    history: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------

    def _init(self, n_inputs: int) -> None:
        rng = np.random.default_rng(self.config.seed)
        h = self.config.hidden
        self.params = {
            "w1": rng.normal(0.0, 0.5 / np.sqrt(n_inputs), (n_inputs, h)),
            "b1": np.zeros(h),
            "w2": rng.normal(0.0, 0.1 / np.sqrt(h), (h, N_ACTIONS)),
            "b2": np.zeros(N_ACTIONS),
        }

    def _scale(self, states: np.ndarray) -> np.ndarray:
        return (states - self.mean_) / self.scale_

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        log_probs, _ = policy_log_probs(self.params, self._scale(states))
        return np.exp(log_probs)

    def greedy_actions(self, states: np.ndarray) -> np.ndarray:
        return self.action_probs(states).argmax(axis=1)

    def value(self, states: np.ndarray, rewards: np.ndarray) -> dict:
        """How the policy's chosen actions would have done on these games."""
        chosen = self.greedy_actions(states)
        got = rewards[np.arange(len(chosen)), chosen]
        return {
            "mean_reward": float(got.mean()),
            "bet_rate": float((chosen != 0).mean()),
            "expected_reward_of_policy": float((self.action_probs(states) * rewards).sum(axis=1).mean()),
        }

    # -- training -----------------------------------------------------------

    def fit(self, states, rewards, val_states, val_rewards, *, warm_start: bool = False,
            iterations: int | None = None) -> "GRPOAgent":
        """Train on (states, rewards), keeping the round that did best on validation.

        With ``warm_start`` the current policy is the starting point and the
        first reference policy, which is how the agent keeps learning as new
        results come in.
        """
        config = self.config
        rng = np.random.default_rng(config.seed + len(self.history))
        states, rewards = np.asarray(states, dtype=float), np.asarray(rewards, dtype=float)
        if not warm_start or self.params is None:
            self.mean_ = states.mean(axis=0)
            self.scale_ = states.std(axis=0) + 1e-8
            self._init(states.shape[1])
            self.history = []
        x = self._scale(states)
        optimizer = _Adam(self.params, config.learning_rate)

        def validation_score() -> float:
            return self.value(val_states, val_rewards)["mean_reward"]

        best_score = validation_score()
        best_params = {k: v.copy() for k, v in self.params.items()}
        stale = 0
        for iteration in range(iterations or config.iterations):
            log_old, _ = policy_log_probs(self.params, x)
            actions = sample_actions(np.exp(log_old), config.group_size, rng)
            sampled_rewards = np.take_along_axis(rewards, actions, 1)
            advantages = group_relative_advantages(sampled_rewards)
            loss = float("nan")
            for _ in range(config.epochs):
                order = rng.permutation(len(x))
                for start in range(0, len(x), config.batch_size):
                    rows = order[start:start + config.batch_size]
                    loss, grads = grpo_loss_and_grads(
                        self.params, x[rows], actions[rows], advantages[rows], log_old[rows], config)
                    optimizer.step(self.params, grads)
            score = validation_score()
            train_value = self.value(states, rewards)
            self.history.append({
                "round": len(self.history) + 1, "loss": loss, "validation_reward": score,
                "train_reward": train_value["mean_reward"], "train_bet_rate": train_value["bet_rate"],
            })
            if score > best_score + 1e-9:
                best_score, stale = score, 0
                best_params = {k: v.copy() for k, v in self.params.items()}
            else:
                stale += 1
                if stale >= config.patience:
                    break
        self.params = best_params
        self.info = {
            "config": asdict(config), "actions": [list(a) for a in ACTIONS], "state_names": list(STATE_NAMES),
            "training_games": int(len(states)), "validation_games": int(len(val_states)),
            "best_validation_reward": float(best_score), "rounds_run": len(self.history),
        }
        return self

    # -- using the agent ----------------------------------------------------

    def decide(self, p_model, home_dec, away_dec) -> pd.DataFrame:
        """The agent's choice for each game: side, stake and how sure the policy is."""
        states = state_matrix(p_model, home_dec, away_dec)
        probs = self.action_probs(states)
        chosen = probs.argmax(axis=1)
        side = np.array([ACTIONS[a][0] for a in chosen])
        stake = np.array([ACTIONS[a][1] for a in chosen])
        price = np.where(side == "home", home_dec, np.where(side == "away", away_dec, np.nan))
        return pd.DataFrame({
            "side": side, "stake": stake, "price": price,
            "policy_confidence": probs.max(axis=1), "pass_chance": probs[:, 0],
        })

    def save(self, folder: Path, name: str = "grpo_agent") -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(folder: Path, name: str = "grpo_agent") -> "GRPOAgent":
        path = folder / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"no saved agent at {path}. Run `mlb train` first.")
        return joblib.load(path)
