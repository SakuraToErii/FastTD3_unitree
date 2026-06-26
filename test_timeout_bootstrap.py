import torch

from fast_td3.fast_td3_utils import (
    resolve_next_observations,
    timeout_bootstrap_observations,
)


def test_timeout_bootstrap_observations_uses_current_obs_only_for_truncations():
    obs = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    next_obs = torch.tensor([[101.0, 110.0], [102.0, 120.0], [103.0, 130.0]])
    truncations = torch.tensor([False, True, False])

    out = timeout_bootstrap_observations(obs, next_obs, truncations)

    torch.testing.assert_close(
        out,
        torch.tensor([[101.0, 110.0], [2.0, 20.0], [103.0, 130.0]]),
    )


def test_timeout_bootstrap_observations_broadcasts_over_history_dims():
    obs = torch.arange(24.0).reshape(3, 2, 4)
    next_obs = obs + 100.0
    truncations = torch.tensor([True, False, True])

    out = timeout_bootstrap_observations(obs, next_obs, truncations)

    torch.testing.assert_close(out[0], obs[0])
    torch.testing.assert_close(out[1], next_obs[1])
    torch.testing.assert_close(out[2], obs[2])


def test_resolve_next_observations_falls_back_when_no_terminal_obs():
    """Without terminal obs, behavior matches the timeout-bootstrap approximation."""
    obs = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    next_obs = torch.tensor([[101.0, 110.0], [102.0, 120.0], [103.0, 130.0]])
    truncations = torch.tensor([False, True, False])
    dones = torch.tensor([0, 1, 0])

    out = resolve_next_observations(obs, next_obs, truncations, dones, terminal_obs=None)

    torch.testing.assert_close(
        out,
        torch.tensor([[101.0, 110.0], [2.0, 20.0], [103.0, 130.0]]),
    )


def test_resolve_next_observations_prefers_terminal_obs_at_done_positions():
    """Done rows are overwritten with the true terminal obs; non-done rows keep
    the approximation (timeout envs reuse pre-step obs, surviving envs use next_obs)."""
    obs = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    next_obs = torch.tensor(
        [[101.0, 110.0], [102.0, 120.0], [103.0, 130.0], [104.0, 140.0]]
    )
    # env 1: termination (done, not truncation) -> terminal obs wins
    # env 2: timeout (truncation, done) -> terminal obs wins over the pre-step fallback
    # env 3: surviving -> next_obs
    truncations = torch.tensor([False, False, True, False])
    dones = torch.tensor([1, 1, 1, 0])
    # terminal obs only has rows for the three done envs, in done order
    terminal_obs = torch.tensor([[201.0, 210.0], [202.0, 220.0], [203.0, 230.0]])

    out = resolve_next_observations(obs, next_obs, truncations, dones, terminal_obs)

    torch.testing.assert_close(
        out,
        torch.tensor(
            [[201.0, 210.0], [202.0, 220.0], [203.0, 230.0], [104.0, 140.0]]
        ),
    )


def test_resolve_next_observations_no_done_keeps_approximation():
    """When nothing is done, terminal rows would be empty and the approximation holds."""
    obs = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    next_obs = torch.tensor([[101.0, 110.0], [102.0, 120.0]])
    truncations = torch.tensor([False, False])
    dones = torch.tensor([0, 0])
    terminal_obs = torch.empty((0, 2))

    out = resolve_next_observations(obs, next_obs, truncations, dones, terminal_obs)

    torch.testing.assert_close(out, next_obs)


if __name__ == "__main__":
    test_timeout_bootstrap_observations_uses_current_obs_only_for_truncations()
    test_timeout_bootstrap_observations_broadcasts_over_history_dims()
    test_resolve_next_observations_falls_back_when_no_terminal_obs()
    test_resolve_next_observations_prefers_terminal_obs_at_done_positions()
    test_resolve_next_observations_no_done_keeps_approximation()
