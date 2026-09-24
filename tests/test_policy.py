from __future__ import annotations

import torch

from shipgame_ai import ActorCritic, NavalEnv


def test_policy_only_selects_masked_actions() -> None:
    env = NavalEnv()
    observation = env.reset(seed=8)
    model = ActorCritic(env.dimension, env.action_count)
    mask = torch.as_tensor(observation.mask()).unsqueeze(0)
    mask[0, :5] = False
    observations = torch.as_tensor(observation.vector()).unsqueeze(0)
    distribution = model.distribution(observations, mask)
    probabilities = distribution.probs[0]
    allowed = torch.as_tensor(observation.action_mask)
    allowed[:5] = False

    assert torch.allclose(
        probabilities[~allowed],
        torch.zeros_like(probabilities[~allowed]),
        atol=1e-6,
    )
    assert torch.isfinite(distribution.entropy()).all()


def test_policy_evaluation_has_expected_shapes() -> None:
    env = NavalEnv()
    observation = env.reset(seed=9)
    model = ActorCritic(env.dimension, env.action_count)
    observations = torch.as_tensor(observation.vector()).unsqueeze(0)
    masks = torch.as_tensor(observation.mask()).unsqueeze(0)
    action, log_probability, entropy = model.act(observations, masks)
    values = model(observations)[1]

    assert action.shape == (1,)
    assert log_probability.shape == (1,)
    assert entropy.shape == (1,)
    assert values.shape == (1,)
