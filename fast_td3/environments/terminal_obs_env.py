"""ManagerBasedRLEnv subclass that captures true terminal observations.

The stock :class:`isaaclab.envs.ManagerBasedRLEnv.step` resets terminated/timed-out
sub-environments *before* computing the final observations, so the observation it
returns for done envs is the *reset* observation of the next episode rather than the
true end-of-episode state ``s_{t+1}``. FastTD3 bootstraps timeout transitions, so using
the reset observation (or the pre-step observation, which is the previous approximation)
as ``next_obs`` leaks next-episode state into the TD target.

``FastTD3ManagerBasedRLEnv`` copies ``ManagerBasedRLEnv.step`` verbatim and inserts a
single hook: right after reward/termination computation and *before* ``_reset_idx``,
it computes the observation group with ``update_history=True`` (so the current step's
reading is appended to the observation history, which matters for ``history_length > 0``
e.g. the G1 velocity task), clones it, and stashes the clone in
``self.extras["terminal_observations"]``. The observation-history circular buffers are
snapshotted and restored around this extra compute so non-reset envs are not
double-appended (their final ``compute(update_history=True)`` at the end of ``step``
still appends exactly once).

This is intentionally a local, explicit copy of ``step`` rather than a monkey-patch so
the control flow is obvious. The trade-off: if IsaacLab reorders the ``step`` pipeline
upstream, this method must be re-aligned once. The body mirrors
``ManagerBasedRLEnv.step`` from the IsaacLab checkout pinned for this repo.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class FastTD3ManagerBasedRLEnv(ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` that exposes true terminal observations via extras."""

    def step(self, action: torch.Tensor):
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set action into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # FastTD3: capture the true end-of-episode observation BEFORE _reset_idx
            # clears the observation history. Done with update_history=True so the
            # current step's reading enters the history (needed for history_length>0).
            self.extras["terminal_observations"] = self._compute_terminal_observations()
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)
        else:
            # Clear any stale capture from a previous step (extras dict is reused).
            self.extras["terminal_observations"] = None

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

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

    def _compute_terminal_observations(self):
        """Compute and clone the observation groups at the terminal state.

        Returns a dict mapping each observation group name (e.g. ``"policy"``,
        ``"critic"``) to a cloned ``(num_envs, *obs_dim)`` tensor for *all* envs. The
        caller (the FastTD3 env wrapper) selects the done-env rows. Surviving envs'
        rows are identical to the observations the final ``compute`` produces and are
        simply discarded by the caller.
        """
        snapshot = self._snapshot_observation_history()
        obs = self.observation_manager.compute(update_history=True)
        terminal = {group_name: value.clone() for group_name, value in obs.items()}
        self._restore_observation_history(snapshot)
        return terminal
