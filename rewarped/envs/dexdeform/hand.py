import os
from glob import glob

import numpy as np
import torch
from gym import spaces
from omegaconf import OmegaConf

import warp as wp

from ...environment import IntegratorType, run_env
from ...mpm_warp_env_mixin import MPMWarpEnvMixin
from ...warp_env import WarpEnv
from .utils.interface import DEFAULT_INITIAL_QPOS
from .utils.io import load_demo

TASK_LENGTHS = {
    # "folding": 250,
    # "rope": 250,
    # "bun": 250,
    # "dumpling": 250,
    # "wrap": 500,
    "flip": 500,
    "lift": 100,
}


class Hand(MPMWarpEnvMixin, WarpEnv):
    sim_name = "Hand" + "DexDeform"
    env_offset = (1.0, 0.0, 0.0)
    env_offset_correction = False

    eval_fk = True
    kinematic_fk = True
    eval_ik = False

    integrator_type = IntegratorType.MPM
    sim_substeps_mpm = 40

    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "body_q") + ("mpm_x", "mpm_v", "mpm_C", "mpm_F_trial", "mpm_F", "mpm_stress")
    control_tensors_names = ("joint_act",)

    TRAJ_OPT = True
    LOAD_DEMO = False

    def __init__(self, task_name="flip", num_envs=2, episode_length=-1, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 24
        if episode_length == -1:
            episode_length = TASK_LENGTHS[task_name]
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.task_name = task_name
        self.action_scale = (0.33 * 0.002) * self.sim_substeps_mpm
        self.downsample_particle = 250

        self.dexdeform_cfg = self.create_cfg_dexdeform()
        print(self.dexdeform_cfg)

    @property
    def observation_space(self):
        d = {
            "particle_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.downsample_particle, 3), dtype=np.float32),
            "com_q": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "joint_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_act,), dtype=np.float32),
        }
        d = spaces.Dict(d)
        return d

    def create_cfg_dexdeform(self):
        cfg_file = os.path.join(os.path.dirname(__file__), f"env_cfgs/{self.task_name}.yml")
        dexdeform_cfg = OmegaConf.load(open(cfg_file, "r"))

        dexdeform_cfg.SIMULATOR.gravity = eval(dexdeform_cfg.SIMULATOR.gravity)
        for i, shape in enumerate(dexdeform_cfg.SHAPES):
            dexdeform_cfg.SHAPES[i].init_pos = eval(shape.init_pos)
            if "width" in shape:
                dexdeform_cfg.SHAPES[i].width = eval(shape.width)
        for i, manipulator in enumerate(dexdeform_cfg.MANIPULATORS):
            dexdeform_cfg.MANIPULATORS[i].init_pos = eval(manipulator.init_pos)
            dexdeform_cfg.MANIPULATORS[i].init_rot = eval(manipulator.init_rot)

        return dexdeform_cfg

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    def create_builder(self):
        builder = super().create_builder()
        self.create_builder_mpm(builder)
        return builder

    def create_env(self, builder):
        mode = self.dexdeform_cfg.SIMULATOR.mode
        if mode == "dual":
            raise NotImplementedError

        for hand_cfg in self.dexdeform_cfg.MANIPULATORS:
            scale = self.dexdeform_cfg.SIMULATOR.scale
            fixed_base = self.dexdeform_cfg.SIMULATOR.get("fixed_base", False)
            self.create_hand(builder, mode, hand_cfg, scale, fixed_base)

    def create_hand(self, builder, mode, hand_cfg, scale, fixed_base):
        init_pos, init_rot, init_qpos = hand_cfg.init_pos, hand_cfg.init_rot, hand_cfg.init_qpos
        init_rot[0] += -np.pi / 2
        if mode == "lh":
            asset_file = "left_hand.xml"
        elif mode == "rh":
            asset_file = "right_hand.xml"
        else:
            raise ValueError

        xform = wp.transform(init_pos, wp.quat_rpy(*init_rot))
        wp.sim.parse_mjcf(
            os.path.join(self.asset_dir, f"shadow/{asset_file}"),
            builder,
            xform=xform,
            floating=not fixed_base,
            stiffness=1000.0,
            damping=0.0,
            # parse_meshes=False,
            # ignore_names=["C_forearm"],
            visual_classes=["D_Vizual"],
            collider_classes=["DC_Hand"],
            enable_self_collisions=False,
            scale=scale,
            reduce_capsule_height_by_radius=True,
            collapse_fixed_joints=True,
            # force_show_colliders=True,
            # hide_visuals=True,
        )

        if init_qpos == "default":
            for i, joint_name in enumerate(builder.joint_name):
                builder.joint_q[i] = DEFAULT_INITIAL_QPOS[f"robot0:{joint_name}"]
        elif init_qpos == "zero":
            for i, joint_name in enumerate(builder.joint_name):
                builder.joint_q[i] = 0.0
        else:
            raise ValueError

    def create_cfg_mpm(self, mpm_cfg):
        mpm_cfg.update(["--physics.sim", "dexdeform"])
        mpm_cfg.update(["--physics.env", "plasticine"])

        # https://github.com/sizhe-li/DexDeform/blob/72f5087ed4e46cf88092f36d6ced9a72978d8c01/mpm/simulator.py#L385
        gravity = np.array(self.dexdeform_cfg.SIMULATOR.gravity) * 30
        mpm_cfg.update(["--physics.sim.gravity", str(tuple(gravity))])
        mpm_cfg.update(["--physics.sim.body_friction", str(self.dexdeform_cfg.SIMULATOR.hand_friction)])
        mpm_cfg.update(["--physics.sim.ground_friction", str(self.dexdeform_cfg.SIMULATOR.ground_friction)])

        def update_plasticine_params():
            E, nu = self.dexdeform_cfg.SIMULATOR.E, self.dexdeform_cfg.SIMULATOR.nu
            yield_stress = self.dexdeform_cfg.SIMULATOR.yield_stress
            rho = 1.0
            mpm_cfg.update(["--physics.env.physics", "plb_plasticine"])
            mpm_cfg.update(["--physics.env.physics.E", str(E)])
            mpm_cfg.update(["--physics.env.physics.nu", str(nu)])
            mpm_cfg.update(["--physics.env.physics.yield_stress", str(yield_stress)])
            mpm_cfg.update(["--physics.env.rho", str(rho)])

        if self.task_name == "lift":
            # update_plasticine_params()

            shape_cfg = self.dexdeform_cfg.SHAPES[0]
            init_pos, width = shape_cfg.init_pos, shape_cfg.width
            resolution = 20
            mpm_cfg.update(["--physics.env.shape", "cube"])
            mpm_cfg.update(["--physics.env.shape.center", str(tuple(init_pos))])
            mpm_cfg.update(["--physics.env.shape.size", str(tuple(width))])
            mpm_cfg.update(["--physics.env.shape.resolution", str(resolution)])
            # TODO: change resolution based on num_particles
        elif self.task_name == "flip":
            update_plasticine_params()

            mpm_cfg.update(["--physics.env.shape", "cylinder_dexdeform"])
            # mpm_cfg.update(["--physics.env.shape.num_particles", str(self.dexdeform_cfg.SIMULATOR.n_particles)])

            # lowering to decrease memory requirements for parallel envs
            mpm_cfg.update(["--physics.sim.num_grids", str(48)])
            mpm_cfg.update(["--physics.env.shape.num_particles", str(2500)])
        else:
            raise NotImplementedError

        print(mpm_cfg)
        return mpm_cfg

    def create_model(self):
        model = super().create_model()
        self.create_model_mpm(model)
        return model

    def init_sim(self):
        self.init_sim_mpm()
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

            self.joint_limit_lower = wp.to_torch(self.model.joint_limit_lower).view(self.num_envs, -1).clone()
            self.joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, -1).clone()

            self.init_dist = torch.ones(self.num_envs, device=self.device, dtype=torch.float)

            if self.task_name == "lift":
                pass
            elif self.task_name == "flip":
                cylinder_indices = torch.arange(0, self.downsample_particle, device=self.device)
                N = self.downsample_particle // 2
                self.half0_indices, self.half1_indices = cylinder_indices[:N], cylinder_indices[N:]
                self.half1_indices = torch.flip(self.half1_indices, [0])

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        with torch.no_grad():
            self.init_dist[env_ids] = 1.0

    @torch.no_grad()
    def randomize_init(self, env_ids):
        if self.task_name == "lift":
            pass
        elif self.task_name == "flip":
            mpm_x = self.state.mpm_x.view(self.num_envs, -1, 3)
            bounds = torch.tensor([0.04, 0.02, 0.04], device=self.device)
            mpm_x[env_ids, :, :] += bounds * (torch.rand(size=(len(env_ids), 1, 3), device=self.device) - 0.5) * 2.0
        else:
            raise NotImplementedError

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        if self.kinematic_fk:
            # ensure joint limit
            joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
            joint_q = joint_q.detach()
            acts = torch.clip(acts, self.joint_limit_lower - joint_q, self.joint_limit_upper - joint_q)

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", acts.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
            self.control.assign("joint_act", joint_act.flatten())

    def compute_observations(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
        particle_q = self.state.mpm_x.clone().view(self.num_envs, -1, 3)

        particle_q -= self.env_offsets.view(self.num_envs, 1, 3)

        if self.downsample_particle is not None:
            num_full = particle_q.shape[1]
            downsample = num_full // self.downsample_particle
            particle_q = particle_q[:, ::downsample, :]
            # assert particle_q.shape[1] == self.downsample_particle

        com_q = particle_q.mean(1)

        self.obs_buf = {
            "joint_q": joint_q,
            "particle_q": particle_q,
            "com_q": com_q,
        }

    def compute_reward(self):
        particle_q = self.obs_buf["particle_q"]
        com_q = self.obs_buf["com_q"]

        if self.task_name == "lift":
            rew = com_q[:, 1]  # maximize height
        elif self.task_name == "flip":
            half0_points, half1_points = particle_q[:, self.half0_indices, :], particle_q[:, self.half1_indices, :]
            dist = torch.linalg.norm(half0_points - half1_points, dim=-1).mean(dim=-1)

            first_mask = self.progress_buf == 1
            if first_mask.any():
                with torch.no_grad():
                    self.init_dist[first_mask] = dist[first_mask]

            ni = (self.init_dist - dist) / self.init_dist
            ni = torch.clamp(ni, -1.0, 1.0)

            rew = ni
        else:
            raise NotImplementedError

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt
                self.renderer.begin_frame(self.render_time)
                # render state 1 (swapped with state 0 just before)
                self.renderer.render(state or self.state_1)
                self.render_mpm(state=state)
                self.renderer.end_frame()

    def render_mpm(self, state=None):
        state = state or self.state_1

        # render mpm particles
        particle_q = state.mpm_x
        if isinstance(particle_q, torch.Tensor):
            particle_q = particle_q.detach().cpu().numpy()
        else:
            particle_q = particle_q.numpy()
        particle_radius = 7.5e-3
        particle_color = (0.875, 0.451, 1.0)  # 0xdf73ff
        self.renderer.render_points("particle_q", particle_q, radius=particle_radius, colors=particle_color)

    def get_demo_actions(self, demo_file):
        demo = load_demo(demo_file)
        demo_traj = []
        for t in range(len(demo["states"]) - 1):
            state = demo["states"][t]
            next_state = demo["states"][t + 1]
            # action = demo["actions"][t]  # joint velocities
            action = (next_state[24][0] - state[24][0]) / self.action_scale
            action = action[None, ...].repeat(self.num_envs, axis=0)
            action = torch.tensor(action, device=self.device, requires_grad=True)
            demo_traj.append(action)
        if len(demo_traj) != self.episode_length:
            print(f"Demo {demo_file} has length {len(demo_traj)}")
            if len(demo_traj) > self.episode_length:
                demo_traj = demo_traj[: self.episode_length]
            elif len(demo_traj) < self.episode_length:
                for t in range(len(demo["states"]) - 1, self.episode_length):
                    action = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=True)
                    demo_traj.append(action)
            assert len(demo_traj) == self.episode_length
        return demo_traj

    def run(self):
        self.init()
        self.initialized = True

        if self.LOAD_DEMO:
            # load actions from DexDeform teleop demos
            demo_dir = os.path.join(self.asset_dir, "../data/DexDeform/demos")
            demo_files = sorted(glob(os.path.join(demo_dir, f"{self.task_name}/*.pkl")))
            demo_file = demo_files[-1]
            traj = self.get_demo_actions(demo_file)
        else:
            traj = [
                # torch.rand(self.num_envs, self.num_actions, device=self.device, requires_grad=True)
                torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=True)
                for _ in range(self.episode_length)
            ]

        train_iters = 50
        train_rate = 0.1
        opt = torch.optim.Adam(traj, lr=train_rate)

        self.iter = 0
        self.max_iter = train_iters
        while self.iter < self.max_iter:
            obs = self.reset(clear_grad=True)

            profiler = {}
            with wp.ScopedTimer("episode", detailed=False, print=False, active=True, dict=profiler):
                obses, actions, rewards, dones, infos = [obs], [], [], [], []
                for i in range(self.episode_length):
                    action = traj[i]
                    obs, reward, done, info = self.step(action)

                    obses.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)

                if self.TRAJ_OPT:
                    actions = torch.stack(actions)
                    rewards = torch.stack(rewards)

                    loss = -rewards[-1]

                    opt.zero_grad()
                    loss.sum().backward()
                    opt.step()

                    grad_norm = [x.grad.norm() for x in traj]
                    grad_norm = torch.stack(grad_norm).mean()

                    print(f"Iter: {self.iter} Loss: {loss.tolist()}")
                    print(f"Grads: {grad_norm.item()}")
                    print(
                        "Traj actions:",
                        actions.mean().item(),
                        actions.std().item(),
                        actions.min().item(),
                        actions.max().item(),
                    )
                else:
                    self.iter = self.max_iter

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
    run_env(Hand, no_grad=False)
