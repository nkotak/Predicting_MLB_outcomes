"""The GRPO agent: its maths is right and it really learns."""

import numpy as np
import pytest

from mlb_predict import agent as ag
from mlb_predict.agent import GRPOAgent, GRPOConfig


def _market(n, seed, *, market_noise, model_noise, vig=0.04):
    """A made-up season where the true win chance of every game is known."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.35, 0.65, n)
    market = np.clip(truth + rng.normal(0, market_noise, n), 0.2, 0.8)
    model = np.clip(truth + rng.normal(0, model_noise, n), 0.2, 0.8)
    home_dec = 1.0 / (market * (1 + vig))
    away_dec = 1.0 / ((1 - market) * (1 + vig))
    home_win = (rng.random(n) < truth).astype(float)
    states = ag.state_matrix(model, home_dec, away_dec)
    rewards = ag.reward_table(home_dec, away_dec, home_win)
    return states, rewards, (truth, home_dec, away_dec, home_win)


def test_reward_table_matches_hand_calculation():
    table = ag.reward_table([2.5], [1.6], [1.0])
    assert table[0, 0] == 0.0  # pass
    home_small = ag.ACTIONS.index(("home", 0.005))
    away_large = ag.ACTIONS.index(("away", 0.02))
    assert table[0, home_small] == pytest.approx(100 * np.log(1 + 0.005 * 1.5))
    assert table[0, away_large] == pytest.approx(100 * np.log(1 - 0.02))


def test_rewards_are_not_all_zero():
    # The earlier version of this project computed a reward that was always zero.
    _, rewards, _ = _market(500, 1, market_noise=0.05, model_noise=0.0)
    assert np.abs(rewards[:, 1:]).min() > 0
    sampled = rewards[:, [0, 1, 4, 6]]
    adv = ag.group_relative_advantages(sampled)
    assert adv.std() > 0.5
    assert np.allclose(adv.mean(axis=1), 0.0)  # centred inside each game's group


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    n, d, g = 40, len(ag.STATE_NAMES), 6
    config = GRPOConfig(hidden=5, clip=0.2, kl_weight=0.3, entropy_weight=0.05, weight_decay=1e-3)
    x = rng.normal(size=(n, d))
    params = {"w1": rng.normal(0, 0.5, (d, 5)), "b1": rng.normal(0, 0.1, 5),
              "w2": rng.normal(0, 0.5, (5, ag.N_ACTIONS)), "b2": rng.normal(0, 0.1, ag.N_ACTIONS)}
    old = {k: v + rng.normal(0, 0.05, v.shape) for k, v in params.items()}
    log_old, _ = ag.policy_log_probs(old, x)
    actions = ag.sample_actions(np.exp(log_old), g, rng)
    advantages = rng.normal(size=(n, g))
    _, grads = ag.grpo_loss_and_grads(params, x, actions, advantages, log_old, config)
    step = 1e-6
    for key in params:
        flat = params[key].ravel()
        for index in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            keep = flat[index]
            flat[index] = keep + step
            up, _ = ag.grpo_loss_and_grads(params, x, actions, advantages, log_old, config)
            flat[index] = keep - step
            down, _ = ag.grpo_loss_and_grads(params, x, actions, advantages, log_old, config)
            flat[index] = keep
            numeric = (up - down) / (2 * step)
            assert grads[key].ravel()[index] == pytest.approx(numeric, rel=1e-4, abs=1e-7), key


def test_clipping_stops_large_policy_moves():
    rng = np.random.default_rng(2)
    config = GRPOConfig(hidden=4, clip=0.2, kl_weight=0.0, weight_decay=0.0)
    x = rng.normal(size=(30, len(ag.STATE_NAMES)))
    params = {"w1": rng.normal(0, 0.5, (x.shape[1], 4)), "b1": np.zeros(4),
              "w2": rng.normal(0, 0.5, (4, ag.N_ACTIONS)), "b2": np.zeros(ag.N_ACTIONS)}
    log_new, _ = ag.policy_log_probs(params, x)
    actions = log_new.argmax(axis=1)[:, None]
    # A reference policy that gave these actions far less chance: ratio is way above 1 + clip.
    log_old = log_new - 2.0 * (np.arange(ag.N_ACTIONS)[None, :] == actions)
    _, grads = ag.grpo_loss_and_grads(params, x, actions, np.ones((30, 1)), log_old, config)
    assert all(np.allclose(g, 0.0) for g in grads.values())


def test_agent_learns_to_exploit_a_known_edge():
    # The model knows the truth and the market is noisy, so real edges exist.
    config = GRPOConfig(iterations=60, patience=60, seed=1)
    train_s, train_r, _ = _market(20000, 10, market_noise=0.07, model_noise=0.0)
    val_s, val_r, _ = _market(6000, 11, market_noise=0.07, model_noise=0.0)
    test_s, test_r, (truth, home_dec, away_dec, _) = _market(20000, 12, market_noise=0.07, model_noise=0.0)
    agent = GRPOAgent(config).fit(train_s, train_r, val_s, val_r)
    chosen = agent.greedy_actions(test_s)
    bets = chosen != 0
    assert bets.mean() > 0.2, "the agent should place bets when there is an edge to take"
    # True expected profit per unit staked of what it chose, known because the truth is known.
    side_home = np.isin(chosen, [1, 2, 3])
    true_edge = np.where(side_home, truth * home_dec - 1, (1 - truth) * away_dec - 1)[bets]
    assert true_edge.mean() > 0.03
    assert (true_edge > 0).mean() > 0.8, "most of its bets should be on the side with a real edge"
    assert agent.value(test_s, test_r)["mean_reward"] > 0
    first, last = agent.history[0], agent.history[-1]
    assert last["train_reward"] > first["train_reward"], "reward should improve with training"


def test_agent_learns_to_pass_when_the_market_is_right():
    # The market knows the truth and the model is noisy, so every bet loses the bookmaker's cut.
    config = GRPOConfig(iterations=60, patience=60, seed=1)
    train_s, train_r, _ = _market(20000, 20, market_noise=0.0, model_noise=0.06)
    val_s, val_r, _ = _market(6000, 21, market_noise=0.0, model_noise=0.06)
    test_s, _, _ = _market(10000, 22, market_noise=0.0, model_noise=0.06)
    agent = GRPOAgent(config).fit(train_s, train_r, val_s, val_r)
    assert (agent.greedy_actions(test_s) == 0).mean() > 0.9
    assert agent.info["best_validation_reward"] >= 0.0


def test_never_bet_is_kept_when_nothing_is_proven_better():
    # Pure noise: the "model" knows nothing and prices are fair plus a cut.
    rng = np.random.default_rng(40)
    config = GRPOConfig(iterations=25, patience=25, seed=2)
    train_s, train_r, _ = _market(4000, 41, market_noise=0.0, model_noise=0.10)
    val_s, val_r, _ = _market(1500, 42, market_noise=0.0, model_noise=0.10)
    agent = GRPOAgent(config).fit(train_s, train_r, val_s, val_r)
    assert agent.info["kept_never_bet_policy"] and agent.info["kept_round"] == 0
    assert (agent.greedy_actions(val_s) == 0).all()
    decisions = agent.decide(rng.uniform(0.4, 0.6, 50), np.full(50, 1.9), np.full(50, 1.9))
    assert (decisions["side"] == "pass").all() and (decisions["stake"] == 0).all()
    assert len(agent.history) > 0, "it still ran its learning rounds"
    # With the safety bar off, the policy it actually learned is still there to inspect.
    raw = agent.decide(rng.uniform(0.3, 0.7, 500), np.full(500, 2.1), np.full(500, 1.8), guarded=False)
    assert set(raw["side"]) <= {"home", "away", "pass"}
    assert not np.array_equal(agent.learned_params["w2"], agent.params["w2"])


def test_warm_start_keeps_learning_and_saves(tmp_path):
    config = GRPOConfig(iterations=10, patience=10, seed=4)
    train_s, train_r, _ = _market(8000, 30, market_noise=0.07, model_noise=0.0)
    val_s, val_r, _ = _market(3000, 31, market_noise=0.07, model_noise=0.0)
    agent = GRPOAgent(config).fit(train_s, train_r, val_s, val_r)
    assert not agent.info["kept_never_bet_policy"]
    rounds = len(agent.history)
    agent.fit(train_s, train_r, val_s, val_r, warm_start=True, iterations=5)
    assert len(agent.history) > rounds
    agent.save(tmp_path)
    again = GRPOAgent.load(tmp_path)
    assert np.array_equal(again.greedy_actions(val_s), agent.greedy_actions(val_s))
