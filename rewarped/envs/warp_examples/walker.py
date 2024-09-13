# from https://github.com/NVIDIA/warp/blob/release-1.3/warp/examples/optim/example_walker.py

# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Walker
#
# Trains a tetrahedral mesh quadruped to run. Feeds 8 time-varying input
# phases as inputs into a single layer fully connected network with a tanh
# activation function. Interprets the output of the network as tet
# activations, which are fed into the wp.sim soft mesh model. This is
# simulated forward in time and then evaluated based on the center of mass
# momentum of the mesh.
#
###########################################################################

import math
import os

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


class Walker(WarpEnv):
    sim_name = "Walker" + "WarpExamples"
    env_offset = (0.0, 0.0, 10.0)

    eval_fk = False
    eval_ik = False

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 80
    # euler_settings = dict(angular_damping=0.05)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 80
    featherstone_settings = dict(angular_damping=0.05, update_mass_matrix_every=sim_substeps_featherstone)

    frame_dt = 1.0 / 60.0

    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("particle_q", "particle_qd")
    control_tensors_names = ("tet_activations",)

    def __init__(
        self,
        num_envs=8,
        episode_length=300,
        early_termination=False,
        **kwargs,
    ):
        # bear
        full_num_obs = 1986  # model.particle_count
        full_num_act = 5354  # model.tet_count
        particle_q_downsample = 6
        action_scale = 0.3  # activation_strength
        act_downsample = 17  # 17 | 35 | 45 | 51 ... (full_num_act // N + 1) * N - full_num_act == 1
        act_padding = 1

        num_obs = full_num_obs // particle_q_downsample
        num_act = full_num_act // act_downsample
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.particle_q_downsample = particle_q_downsample
        self.action_scale = action_scale
        self.act_downsample = act_downsample
        self.act_padding = act_padding

        self.phase_count = 8
        self.phase_step = 2.0 * math.pi / self.phase_count
        self.phase_freq = 5.0

    @property
    def observation_space(self):
        d = {
            "phase": spaces.Box(low=-np.inf, high=np.inf, shape=(self.phase_count,), dtype=np.float32),
            "loss": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "com_q": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "com_qd": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "particle_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs, 3), dtype=np.float32),
            "particle_qd": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs, 3), dtype=np.float32),
            "actions": spaces.Box(low=-1.0, high=1.0, shape=(self.num_act,), dtype=np.float32),
        }
        d = spaces.Dict(d)
        return d

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.default_particle_radius = 0.05
        return builder

    def create_env(self, builder):
        self.create_bear(builder)

    def create_bear(self, builder):
        asset_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bear.usd"))
        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bear"))
        points = geom.GetPointsAttr().Get()

        xform = geom.ComputeLocalToWorldTransform(0.0)
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        points = [wp.vec3(point) for point in points]
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.5, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
            k_mu=2000.0,
            k_lambda=2000.0,
            k_damp=2.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

    def create_model(self):
        model = super().create_model()

        # bear
        model.soft_contact_ke = 2.0e3
        model.soft_contact_kd = 0.1
        model.soft_contact_kf = 10.0
        model.soft_contact_mu = 0.7

        # # NOTE: unneeded since setting default_particle_radius in create_modelbuilder()
        # radii = wp.zeros(model.particle_count, dtype=float)
        # r = 0.05
        # radii.fill_(r)
        # model.particle_radius = radii

        return model

    def init_sim(self):
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.tet_activations = wp.to_torch(self.model.tet_activations).view(self.num_envs, -1).clone()
            self.tet_activations_indices = ...

    @torch.no_grad()
    def randomize_init(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        if self.act_downsample is not None:
            N = self.model.tet_count // self.num_envs
            acts = torch.cat([acts, torch.zeros(self.num_envs, self.act_padding, device=self.device)], dim=-1)
            acts = torch.repeat_interleave(acts, self.act_downsample, dim=-1)
            acts = acts[:, :N]

        if self.tet_activations_indices is ...:
            self.control.assign("tet_activations", acts.flatten())
        else:
            tet_activations = self.scatter_actions(self.tet_activations, self.tet_activations_indices, acts)
            self.control.assign("tet_activations", tet_activations.flatten())

    def compute_observations(self):
        particle_q = self.state.particle_q.clone().view(self.num_envs, -1, 3)
        particle_qd = self.state.particle_qd.clone().view(self.num_envs, -1, 3)

        particle_q -= self.env_offsets.view(self.num_envs, 1, 3)

        com_q = particle_q.mean(1)
        # compute center of mass velocity
        com_qd = particle_qd.mean(1)

        loss = com_qd[:, 2].pow(2).sqrt() + com_qd[:, 1].pow(2).sqrt() - com_qd[:, 0]  # sqrt(vz**2) + sqrt(vy**2) - vx
        loss = loss.reshape(self.num_envs, 1)

        phase_count, phase_step, phase_freq = (
            self.phase_count,
            self.phase_step,
            self.phase_freq,
        )
        sim_time = self.progress_buf * self.frame_dt
        phase_counts = (
            torch.arange(phase_count, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        phase = torch.sin(phase_freq * sim_time.unsqueeze(-1) + phase_counts * phase_step)

        particle_q = particle_q[:, :: self.particle_q_downsample, :]
        particle_qd = particle_qd[:, :: self.particle_q_downsample, :]

        self.obs_buf = {
            "phase": phase,
            "loss": loss,
            "com_q": com_q,
            "com_qd": com_qd,
            "particle_q": particle_q,
            "particle_qd": particle_qd,
            "actions": self.actions.clone(),
        }

    def compute_reward(self):
        loss = self.obs_buf["loss"].squeeze(-1)
        rew = -loss

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = (
            self.episode_length,
            self.early_termination,
        )
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated

    def run_cfg(self):
        assert self.requires_grad

        train_iters = 30
        train_rate = 0.025

        tet_count = self.num_act
        phase_count, phase_step, phase_freq = (
            self.phase_count,
            self.phase_step,
            self.phase_freq,
        )

        phases = []
        for i in range(self.episode_length):
            phase = torch.zeros(phase_count, dtype=torch.float, device=self.device, requires_grad=True)
            phases.append(phase)

        # single layer linear network
        k = 1.0 / phase_count
        net = nn.Sequential(
            nn.Linear(phase_count, tet_count, bias=True),
            nn.Tanh(),
        )
        weights, bias = net[0].weight, net[0].bias
        rng = np.random.default_rng(42)
        _weights = rng.uniform(-np.sqrt(k), np.sqrt(k), (tet_count, phase_count))
        weights.data = torch.tensor(_weights, dtype=torch.float, device=self.device, requires_grad=True)
        # nn.init.uniform_(weights, -np.sqrt(k), np.sqrt(k))
        nn.init.zeros_(bias)

        self._run_weights = weights

        net.to(self.device)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=train_rate)
        print(net)

        def policy_fn(i, obs):
            sim_time = i * self.frame_dt
            phase_counts = torch.arange(
                phase_count,
                dtype=torch.float,
                device=self.device,
                requires_grad=True,
            )
            phases[i] = torch.sin(phase_freq * sim_time + phase_counts * phase_step)
            phases[i] = phases[i].repeat(self.num_envs, 1)
            action = net(phases[i])
            return action

        return train_iters, opt, policy_fn

    def run_loss(self, traj, opt, policy):
        obses, actions, rewards, dones, infos = traj

        # loss = torch.stack([obs["loss"] for obs in obses[1:]])  # skip the first observation
        # loss = loss.sum(0)  # sum over time
        # # loss /= self.num_envs

        loss = -torch.stack(rewards).sum(0)

        # Optimization step
        opt.zero_grad()
        loss.sum().backward()
        opt.step()

        weights = self._run_weights
        grad_norm = weights.grad.norm()
        actions = torch.stack(actions)

        print(f"Iter: {self.iter} Loss: {loss.tolist()}")
        print(f"Grad W: {grad_norm.item()}")
        print(
            "Actions:",
            actions.mean().item(),
            actions.std().item(),
            actions.min().item(),
            actions.max().item(),
        )


if __name__ == "__main__":
    run_env(Walker, no_grad=False)
