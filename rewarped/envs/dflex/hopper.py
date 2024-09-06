# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os

import torch

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


class Hopper(WarpEnv):
    sim_name = "Hopper" + "Dflex"
    env_offset = (0.0, 0.0, 0.0)

    eval_fk = True
    eval_ik = False

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 16
    # euler_settings = dict(angular_damping=0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    frame_dt = 1.0 / 60.0
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=64, episode_length=1000, early_termination=True, **kwargs):
        num_obs = 11
        num_act = 3
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 200.0
        self.termination_height = -0.45
        self.termination_angle = math.pi / 6.0
        self.termination_height_tolerance = 0.15
        self.termination_angle_tolerance = 0.05
        self.height_rew_scale = 1.0
        self.action_penalty = -1e-1

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    def create_articulation(self, builder):
        # dFlex settings
        wp.sim.parse_mjcf(
            os.path.join(self.asset_dir, "dflex/hopper.xml"),
            builder,
            density=1000.0,
            stiffness=0.0,
            damping=2.0,
            contact_ke=2.0e4,
            contact_kd=1.0e3,
            contact_kf=1.0e3,
            contact_mu=0.9,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            armature=1.0,
            enable_self_collisions=False,
            # up_axis="Y",
        )

        xform = wp.transform([0.0, 0.0, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5))
        builder.joint_X_p[0] = xform

        builder.joint_axis_mode = [wp.sim.JOINT_MODE_FORCE] * len(builder.joint_axis_mode)
        builder.joint_q[3:6] = [0.0, 0.0, 0.0]

    def init_sim(self):
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            # self.joint_act_indices = ...
            joint_act_indices = [3, 4, 5]
            joint_act_indices = torch.tensor(joint_act_indices, device=self.device)
            self.joint_act_indices = joint_act_indices.unsqueeze(0).expand(self.num_envs, -1)

            self.start_joint_q = self.state.joint_q.view(self.num_envs, -1).clone()
            self.start_joint_qd = self.state.joint_qd.view(self.num_envs, -1).clone()

            self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
            self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device)
            self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)

            self.x_unit_tensor = self.x_unit_tensor.repeat((self.num_envs, 1))
            self.y_unit_tensor = self.y_unit_tensor.repeat((self.num_envs, 1))
            self.z_unit_tensor = self.z_unit_tensor.repeat((self.num_envs, 1))

            self.start_pos = self.start_joint_q[:, :3]
            self.start_rotation = torch.tensor([0.0], device=self.device)

            # initialize some data used later on
            # todo - switch to z-up
            self.up_vec = self.y_unit_tensor.clone()

    @torch.no_grad()
    def randomize_init(self, env_ids):
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        N = len(env_ids)
        num_joint_q = 6
        num_joint_qd = 6

        joint_q[env_ids, 0:2] += 0.05 * (torch.rand(size=(N, 2), device=self.device) - 0.5) * 2.0
        joint_q[env_ids, 2] = (torch.rand(N, device=self.device) - 0.5) * 0.1
        joint_q[env_ids, 3:] += 0.05 * (torch.rand(size=(N, num_joint_q - 3), device=self.device) - 0.5) * 2.0
        joint_qd[env_ids, :] = 0.05 * (torch.rand(size=(N, num_joint_qd), device=self.device) - 0.5) * 2.0

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", acts.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
            self.control.assign("joint_act", joint_act.flatten())

    def compute_observations(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.clone().view(self.num_envs, -1)
        self.obs_buf = torch.cat([joint_q, joint_qd], dim=-1)

    def compute_reward(self):
        height_diff = self.obs_buf[:, 0] - (self.termination_height + self.termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, 0.3)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)

        angle_reward = 1.0 * (-(self.obs_buf[:, 1] ** 2) / (self.termination_angle**2) + 1.0)

        progress_reward = self.obs_buf[:, 5]

        rew = progress_reward + height_reward + angle_reward + torch.sum(self.actions**2, dim=-1) * self.action_penalty

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            terminated = self.obs_buf[:, 0] < self.termination_height
            reset = torch.where(terminated, torch.ones_like(reset), reset)
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated


if __name__ == "__main__":
    run_env(Hopper)
