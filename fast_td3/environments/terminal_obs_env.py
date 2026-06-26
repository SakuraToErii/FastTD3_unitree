"""ManagerBasedRLEnv subclass that captures true terminal observations.

The stock :class:`isaaclab.envs.ManagerBasedRLEnv.step` resets terminated/timed-out
sub-environments *before* computing the final observations, so the observation it
returns for done envs is the *reset* observation of the next episode rather than the
true end-of-episode state ``s_{t+1}``. FastTD3 bootstraps timeout transitions, so using
the reset observation (or the pre-step observation, which is the previous approximation)
as ``next_obs`` leaks next-episode state into the TD target.

Capture is hooked at the reset boundary instead of by copying ``step``: ``_reset_idx`` is
the single point where done envs are about to be cleared. Right before delegating to the
real reset we compute the observation with ``update_history=True`` (so the current step's
reading enters the history window -- needed for ``history_length > 0`` e.g. the G1 velocity
task), clone it into ``extras["terminal_observations"]``, then undo the append for surviving
envs via snapshot/restore so their final ``compute(update_history=True)`` at the end of
``step`` still appends exactly once.

Overriding ``_reset_idx`` (rather than re-pasting ``step``) keeps this robust to IsaacLab
reordering the ``step`` pipeline upstream; only ``_reset_idx``'s ``(env_ids)`` signature
matters. The observation captured here is the full ``(num_envs, *)`` tensor for every
group; the FastTD3 env wrapper selects the done-env rows.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class FastTD3ManagerBasedRLEnv(ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` that exposes true terminal observations via extras."""

    def step(self, action: torch.Tensor):
        # ``extras`` persists across steps; clear the previous capture so a step
        # with no resets does not hand the wrapper a stale terminal tensor.
        self.extras["terminal_observations"] = None
        self._capture_terminal_observations = True
        try:
            return super().step(action)
        finally:
            self._capture_terminal_observations = False

    def _reset_idx(self, env_ids):
        if not getattr(self, "_capture_terminal_observations", False):
            super()._reset_idx(env_ids)
            return

        # True end-of-episode observation for envs about to be reset: compute with
        # the current reading appended to history, clone, then undo the append so
        # surviving envs are not double-appended by the final compute in ``step``.
        snapshot = self._snapshot_observation_history()
        try:
            obs = self.observation_manager.compute(update_history=True)
            self.extras["terminal_observations"] = {
                group: value.clone() for group, value in obs.items()
            }
        finally:
            self._restore_observation_history(snapshot)
        super()._reset_idx(env_ids)

    # ------------------------------------------------------------------
    # Terminal observation capture helpers
    # ------------------------------------------------------------------
    def _snapshot_observation_history(self) -> list:
        """Clone the state of every observation-history circular buffer.

        ``observation_manager.compute(update_history=True)`` appends the freshly
        computed reading to each term's circular buffer for *all* envs. We only want
        that append to stick for the envs about to be reset (their history is cleared
        by ``_reset_idx`` anyway). For the surviving envs we must undo the append so
        their subsequent final ``compute(update_history=True)`` appends exactly once.
        """
        snapshot = []
        history_buffers = self.observation_manager._group_obs_term_history_buffer
        for group_name in history_buffers:
            for term_name, cb in history_buffers[group_name].items():
                snapshot.append(
                    (
                        cb,
                        cb._buffer.clone() if cb._buffer is not None else None,
                        cb._pointer,
                        cb._num_pushes.clone(),
                    )
                )
        return snapshot

    def _restore_observation_history(self, snapshot: list) -> None:
        """Restore circular-buffer state captured by :meth:`_snapshot_observation_history`."""
        for cb, buffer_clone, pointer, num_pushes_clone in snapshot:
            if buffer_clone is None:
                cb._buffer = None
            else:
                if cb._buffer is None:
                    cb._buffer = buffer_clone.clone()
                else:
                    cb._buffer.copy_(buffer_clone)
            cb._pointer = pointer
            cb._num_pushes.copy_(num_pushes_clone)
