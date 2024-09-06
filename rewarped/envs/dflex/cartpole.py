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
from .utils.torch_utils import normalize_angle


class Cartpole(WarpEnv):
    sim_name = "Cartpole" + "Dflex"
    env_offset = (0.0, 0.0, 2.5)

    eval_fk = True
    eval_ik = False

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 4
    # euler_settings = dict(angular_damping=0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 4
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    joint_attach_ke = 1600.0
    joint_attach_kd = 20.0

    frame_dt = 1.0 / 60.0
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=64, episode_length=240, early_termination=False, **kwargs):
        num_obs = 5
        num_act = 1
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 1000.0

        # loss related
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1
        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1
        self.cart_action_penalty = 0.0

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    def create_articulation(self, builder):
        contact_params = dict(  # dFlex.urdf_load defaults
            contact_ke=1.0e4,
            contact_kd=1.0e4,
            contact_kf=1.0e2,
            contact_mu=0.25,
            contact_restitution=0.0,
            contact_thickness=0.0,
        )

        # dFlex settings
        wp.sim.parse_urdf(
            os.path.join(self.asset_dir, "dflex/cartpole.urdf"),
            builder,
            xform=wp.transform([0.0, 2.5, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            **contact_params,
            # density=1000.0,
            # stiffness=0.0,
            # damping=0.0,
            # armature=0.0,
            limit_ke=1.0e2,
            limit_kd=1.0,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )

        builder.joint_limit_lower = [-4.0, -1000.0]
        builder.joint_limit_upper = [4.0, 1000.0]
        builder.joint_q = [0.0, -math.pi]

        builder.joint_axis_mode = [wp.sim.JOINT_MODE_FORCE] * len(builder.joint_axis_mode)
        builder.joint_act = [0.0, 0.0]

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            # self.joint_act_indices = ...
            joint_act_indices = [0]
            joint_act_indices = torch.tensor(joint_act_indices, device=self.device)
            self.joint_act_indices = joint_act_indices.unsqueeze(0).expand(self.num_envs, -1)

            self.start_joint_q = wp.to_torch(self.model.joint_q).view(self.num_envs, -1).clone()
            self.start_joint_qd = wp.to_torch(self.model.joint_qd).view(self.num_envs, -1).clone()

    @torch.no_grad()
    def randomize_init(self, env_ids):
        num_joint_q = 2
        num_joint_qd = 2

        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        joint_q[env_ids, :] += math.pi * (torch.rand(size=(len(env_ids), num_joint_q), device=self.device) - 0.5)
        joint_qd[env_ids, :] += 0.5 * (torch.rand(size=(len(env_ids), num_joint_qd), device=self.device) - 0.5)

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

        x = joint_q[:, 0:1]
        theta = joint_q[:, 1:2]
        xdot = joint_qd[:, 0:1]
        theta_dot = joint_qd[:, 1:2]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.obs_buf = torch.cat([x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

    def compute_reward(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)

        x = self.obs_buf[:, 0]
        theta = normalize_angle(joint_q[:, 1])
        xdot = self.obs_buf[:, 1]
        theta_dot = self.obs_buf[:, 4]

        rew = (
            -torch.pow(theta, 2.0) * self.pole_angle_penalty
            - torch.pow(theta_dot, 2.0) * self.pole_velocity_penalty
            - torch.pow(x, 2.0) * self.cart_position_penalty
            - torch.pow(xdot, 2.0) * self.cart_velocity_penalty
            - torch.sum(self.actions**2, dim=-1) * self.cart_action_penalty
        )

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated


if __name__ == "__main__":
    run_env(Cartpole)
