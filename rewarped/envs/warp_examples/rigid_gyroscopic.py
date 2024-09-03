# from https://github.com/NVIDIA/warp/blob/release-1.3/warp/examples/sim/example_rigid_gyroscopic.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Gyroscopic
#
# Demonstrates the Dzhanibekov effect where rigid bodies will tumble in
# free space due to unstable axes of rotation.
#
###########################################################################

import torch

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


class RigidGyroscopic(WarpEnv):
    sim_name = "RigidGyroscopic" + "WarpExamples"
    env_offset = (1.0, 1.0, 0.0)

    eval_fk = True
    eval_ik = False

    integrator_type = IntegratorType.EULER
    sim_substeps_euler = 1
    euler_settings = dict(angular_damping=0.05)

    # integrator_type = IntegratorType.FEATHERSTONE
    # sim_substeps_featherstone = 1
    # featherstone_settings = dict(angular_damping=0.05, update_mass_matrix_every=sim_substeps_featherstone)

    frame_dt = 1.0 / 120.0

    up_axis = "Z"
    ground_plane = False
    gravity = 0.0

    def __init__(self, num_envs=8, episode_length=2000, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 0
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

    def create_env(self, builder):
        self.create_articulation(builder)

    def create_articulation(self, builder):
        self.scale = 0.5

        b = builder.add_body()

        # axis shape
        builder.add_shape_box(
            pos=wp.vec3(0.3 * self.scale, 0.0, 0.0),
            hx=0.25 * self.scale,
            hy=0.1 * self.scale,
            hz=0.1 * self.scale,
            density=100.0,
            body=b,
        )

        # tip shape
        builder.add_shape_box(
            pos=wp.vec3(0.0, 0.0, 0.0),
            hx=0.05 * self.scale,
            hy=0.2 * self.scale,
            hz=1.0 * self.scale,
            density=100.0,
            body=b,
        )

        # initial spin
        builder.body_qd[0] = (25.0, 0.01, 0.01, 0.0, 0.0, 0.0)

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

    def reset_idx(self, env_ids):
        if self.early_termination:
            raise NotImplementedError
        else:
            super().reset_idx(env_ids)

    def randomize_init(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        pass

    def compute_observations(self):
        self.obs_buf = {}

    def compute_reward(self):
        rew = 0.0

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
    run_env(RigidGyroscopic)
