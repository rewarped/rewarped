# from https://github.com/NVIDIA/warp/blob/release-1.3/warp/examples/optim/example_cloth_throw.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Grad Cloth
#
# Shows how to use Warp to optimize the initial velocities of a piece of
# cloth such that its center of mass hits a target after a specified time.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import math

import numpy as np
import torch

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


class ClothThrow(WarpEnv):
    sim_name = "ClothThrow" + "WarpExamples"
    env_offset = (0.0, 0.0, 5.0)

    eval_fk = False
    eval_ik = False

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 16
    # euler_settings = dict(angular_damping=0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    frame_dt = 1.0 / 60.0
    episode_duration = 2.0  # seconds

    up_axis = "Y"
    ground_plane = False

    state_tensors_names = ("particle_q", "particle_qd")

    def __init__(self, num_envs=8, episode_length=-1, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 0

        episode_length = int(self.episode_duration / self.frame_dt)
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 1.0
        self.render_traj = True

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.default_particle_radius = 0.01
        return builder

    def create_env(self, builder):
        self.create_cloth(builder)

    def create_cloth(self, builder):
        dim_x = 16
        dim_y = 16

        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.1, 0.1, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.25),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0 / dim_x,
            cell_y=1.0 / dim_y,
            mass=1.0,
            tri_ke=10000.0,
            tri_ka=10000.0,
            tri_kd=100.0,
            tri_lift=10.0,
            tri_drag=5.0,
        )

    def init_sim(self):
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.target = (8.0, 0.0, 0.0)
            self.target = torch.tensor(self.target, device=self.device)

    def reset_idx(self, env_ids):
        if self.early_termination:
            raise NotImplementedError
        else:
            super().reset_idx(env_ids)
            self.traj_verts = []

    def randomize_init(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        pass

    def compute_observations(self):
        particle_q = self.state.particle_q.clone().view(self.num_envs, -1, 3)
        particle_qd = self.state.particle_qd.clone().view(self.num_envs, -1, 3)

        particle_q -= self.env_offsets.view(self.num_envs, 1, 3)

        com_q = particle_q.mean(1)
        delta = com_q - self.target

        self.obs_buf = {
            "particle_q": particle_q,
            "particle_qd": particle_qd,
            "com_q": com_q,
            "delta": delta,
        }

    def compute_reward(self):
        delta = self.obs_buf["delta"]
        loss = torch.einsum("bi,bi->b", delta, delta)  # dot product
        rew = -1.0 * loss

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

    def render(self, state=None):
        if not self.render_traj:
            super().render(state)
        else:
            # render state 1 (swapped with state 0 just before)
            state = state or self.state_1
            traj_vert = self.state.particle_q.view(self.num_envs, -1, 3).mean(1).tolist()
            self.traj_verts.append(traj_vert)

            if self.renderer is not None:
                with wp.ScopedTimer("render", False):
                    self.render_time += self.frame_dt
                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(state)

                    traj_verts = np.array(self.traj_verts).transpose(1, 0, 2)  # -> B, T, 3
                    for i in range(self.num_envs):
                        if self.progress_buf[i] == 0:
                            continue

                        pos = (self.target + self.env_offsets[i]).tolist()
                        self.renderer.render_box(
                            pos=pos,
                            rot=wp.quat_identity(),
                            extents=(0.1, 0.1, 0.1),
                            color=(1.0, 0.0, 0.0),
                            name=f"target{i}",
                        )

                        if traj_verts.shape[1] > 1:
                            self.renderer.render_line_strip(
                                vertices=traj_verts[i],
                                color=wp.render.bourke_color_map(0.0, getattr(self, "max_iter", 64.0), self.iter),
                                radius=0.02,
                                name=f"iter{self.iter}_traj{i}",
                            )

                    self.renderer.end_frame()

    def run(self):
        self.init()
        self.initialized = True

        state_0 = self.state_0
        particle_q_0, particle_qd_0 = self.state_tensors

        train_iters = 64
        train_rate = 5.0
        opt = torch.optim.SGD([particle_qd_0], lr=train_rate)
        opt.add_param_group({"params": [particle_q_0], "lr": 0})  # frozen params, adding so grads are zeroed

        self.iter = 0
        self.max_iter = train_iters
        while self.iter < self.max_iter:
            obs = self.reset()

            self.state_0 = state_0
            self.state_tensors = [particle_q_0, particle_qd_0]
            self.render_traj = self.iter % 4 == 0

            profiler = {}
            with wp.ScopedTimer("episode", detailed=False, print=False, active=True, dict=profiler):
                obses, actions, rewards, dones, infos = [obs], [], [], [], []
                for i in range(self.episode_length):
                    action = torch.rand((self.num_envs, self.num_actions), device=self.device) * 2.0 - 1.0
                    obs, reward, done, info = self.step(action)

                    obses.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)

                loss = -rewards[-1]

                opt.zero_grad()
                loss.sum().backward()
                opt.step()

                print(f"Iter: {self.iter} Loss: {loss.tolist()}")
                print(f"Grad particle q, qd: {particle_q_0.grad.norm().item()}, {particle_qd_0.grad.norm().item()}")

            avg_time = np.array(profiler["episode"]).mean() / self.episode_length
            avg_steps_second = 1000.0 * float(self.num_envs) / avg_time
            total_time_second = np.array(profiler["episode"]).sum() / 1000.0

            print(
                f"num_envs: {self.num_envs} |",
                f"steps/second: {avg_steps_second:.4} |",
                f"milliseconds/step: {avg_time:.4f} |",
                f"total_seconds: {total_time_second:.4f} |",
            )
            print()

            self.iter += 1

        if self.renderer is not None:
            self.renderer.save()

        return 1000.0 * float(self.num_envs) / avg_time


if __name__ == "__main__":
    run_env(ClothThrow, no_grad=False)
