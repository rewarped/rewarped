from typing import Optional, Any, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from torch import Tensor
import warp as wp
from warp.context import Devicelike

from warp.sim.model import ModelShapeGeometry
from warp.sim.collide import sphere_sdf, sphere_sdf_grad
from warp.sim.collide import box_sdf, box_sdf_grad
from warp.sim.collide import capsule_sdf, capsule_sdf_grad
from warp.sim.collide import cylinder_sdf, cylinder_sdf_grad
from warp.sim.collide import cone_sdf, cone_sdf_grad
from warp.sim.collide import plane_sdf

from .. import config
from ..warp import Tape, CondTape
from .base import Statics, State, Model, ModelBuilder, StateInitializer, StaticsInitializer, ShapeLike
from . import materials
from . import shapes


@wp.struct
class MPMStatics(Statics):

    vol: wp.array(dtype=float)
    rho: wp.array(dtype=float)
    env_id: wp.array(dtype=int)
    material_id: wp.array(dtype=int)
    material: wp.array(dtype=materials.MPMMaterial)

    def init(self, shape: ShapeLike, device: Devicelike = None) -> None:
        self.vol = wp.zeros(shape=shape, dtype=float, device=device, requires_grad=False)
        self.rho = wp.zeros(shape=shape, dtype=float, device=device, requires_grad=False)
        self.env_id = wp.zeros(shape=shape, dtype=int, device=device, requires_grad=False)
        self.material_id = wp.zeros(shape=shape, dtype=int, device=device, requires_grad=False)
        self.material = wp.zeros(shape=(0,), dtype=materials.MPMMaterial, device=device, requires_grad=False)

    def update_vol(self, sections: list[int], vols: list[float]) -> None:
        offset = 0
        for section, vol in zip(sections, vols):
            wp.launch(self.set_float, dim=self.vol.shape, inputs=[self.vol, offset, offset + section, vol], device=self.vol.device)
            offset += section

    def update_rho(self, sections: list[int], rhos: list[float]) -> None:
        offset = 0
        for section, rho in zip(sections, rhos):
            wp.launch(self.set_float, dim=self.rho.shape, inputs=[self.rho, offset, offset + section, rho], device=self.rho.device)
            offset += section

    def update_env_id(self, sections: list[int], env_ids: list[float]) -> None:
        offset = 0
        for section, env_id in zip(sections, env_ids):
            wp.launch(self.set_int, dim=self.env_id.shape, inputs=[
                self.env_id, offset, offset + section, env_id], device=self.env_id.device)
            offset += section

    def update_material(self, sections: list[int], material_ids: list[int], _materials: list[materials.MPMMaterial]) -> None:
        offset = 0
        for section, material_id in zip(sections, material_ids):
            wp.launch(self.set_int, dim=self.material_id.shape, inputs=[
                self.material_id, offset, offset + section, material_id], device=self.material_id.device)
            offset += section

        materials0 = self.material.numpy().tolist()
        _materials = materials0 + _materials
        self.material = wp.array(_materials, dtype=materials.MPMMaterial, device=self.material.device)


@wp.struct
class MPMParticleData(object):

    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    C: wp.array(dtype=wp.mat33)
    stress: wp.array(dtype=wp.mat33)

    F_trial: wp.array(dtype=wp.mat33)
    F: wp.array(dtype=wp.mat33)

    @staticmethod
    @wp.kernel
    def init_F_kernel(F: wp.array(dtype=wp.mat33)) -> None:
        p = wp.tid()
        F[p] = wp.identity(n=3, dtype=float)

    def init_F(self) -> None:
        wp.launch(self.init_F_kernel, dim=self.F_trial.shape, inputs=[self.F_trial], device=self.F_trial.device)
        wp.launch(self.init_F_kernel, dim=self.F.shape, inputs=[self.F], device=self.F.device)

    def init(self, shape: ShapeLike, device: Devicelike = None, requires_grad: bool = False) -> None:
        self.x = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.v = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.C = wp.zeros(shape=shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)
        self.stress = wp.zeros(shape=shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)

        self.F_trial = wp.empty(shape=shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)
        self.F = wp.empty(shape=shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)
        self.init_F()

    def clear(self) -> None:
        self.x.zero_()
        self.v.zero_()
        self.C.zero_()
        self.stress.zero_()

        self.init_F()

    def zero_grad(self) -> None:
        if self.x.requires_grad:
            self.x.grad.zero_()
        if self.v.requires_grad:
            self.v.grad.zero_()
        if self.C.requires_grad:
            self.C.grad.zero_()
        if self.F_trial.requires_grad:
            self.F_trial.grad.zero_()
        if self.F.requires_grad:
            self.F.grad.zero_()
        if self.stress.requires_grad:
            self.stress.grad.zero_()

    def clone(self, requires_grad: Optional[bool] = None) -> 'MPMParticleData':
        clone = MPMParticleData()
        clone.x = wp.clone(self.x, requires_grad=requires_grad)
        clone.v = wp.clone(self.v, requires_grad=requires_grad)
        clone.C = wp.clone(self.C, requires_grad=requires_grad)
        clone.F_trial = wp.clone(self.F_trial, requires_grad=requires_grad)
        clone.F = wp.clone(self.F, requires_grad=requires_grad)
        clone.stress = wp.clone(self.stress, requires_grad=requires_grad)
        return clone

    def zeros(self, requires_grad: Optional[bool] = None) -> 'MPMParticleData':
        zeros = MPMParticleData()
        zeros.x = wp.zeros_like(self.x, requires_grad=requires_grad)
        zeros.v = wp.zeros_like(self.v, requires_grad=requires_grad)
        zeros.C = wp.zeros_like(self.C, requires_grad=requires_grad)
        zeros.F_trial = wp.zeros_like(self.F_trial, requires_grad=requires_grad)
        zeros.F = wp.zeros_like(self.F, requires_grad=requires_grad)
        zeros.init_F()
        zeros.stress = wp.zeros_like(self.stress, requires_grad=requires_grad)
        return zeros

    def empty(self, requires_grad: Optional[bool] = None) -> 'MPMParticleData':
        empty = MPMParticleData()
        empty.x = wp.empty_like(self.x, requires_grad=requires_grad)
        empty.v = wp.empty_like(self.v, requires_grad=requires_grad)
        empty.C = wp.empty_like(self.C, requires_grad=requires_grad)
        empty.F_trial = wp.empty_like(self.F_trial, requires_grad=requires_grad)
        empty.F = wp.empty_like(self.F, requires_grad=requires_grad)
        # empty.init_F()
        empty.stress = wp.empty_like(self.stress, requires_grad=requires_grad)
        return empty


@wp.struct
class MPMGridData(object):

    v: wp.array(dtype=wp.vec3, ndim=4)
    mv: wp.array(dtype=wp.vec3, ndim=4)
    m: wp.array(dtype=float, ndim=4)

    def init(self, shape: ShapeLike, device: Devicelike = None, requires_grad: bool = False) -> None:

        self.v = wp.zeros(shape=shape, dtype=wp.vec3, ndim=4, device=device, requires_grad=requires_grad)
        self.mv = wp.zeros(shape=shape, dtype=wp.vec3, ndim=4, device=device, requires_grad=requires_grad)
        self.m = wp.zeros(shape=shape, dtype=float, ndim=4, device=device, requires_grad=requires_grad)

    def clear(self) -> None:
        self.v.zero_()
        self.mv.zero_()
        self.m.zero_()

    def zero_grad(self) -> None:
        if self.v.requires_grad:
            self.v.grad.zero_()
        if self.mv.requires_grad:
            self.mv.grad.zero_()
        if self.m.requires_grad:
            self.m.grad.zero_()

    def clone(self, requires_grad: Optional[bool] = None) -> 'MPMGridData':
        clone = MPMGridData()
        clone.v = wp.clone(self.v, requires_grad=requires_grad)
        clone.mv = wp.clone(self.mv, requires_grad=requires_grad)
        clone.m = wp.clone(self.m, requires_grad=requires_grad)
        return clone

    def zeros(self, requires_grad: Optional[bool] = None) -> 'MPMGridData':
        zeros = MPMGridData()
        zeros.v = wp.zeros_like(self.v, requires_grad=requires_grad)
        zeros.mv = wp.zeros_like(self.mv, requires_grad=requires_grad)
        zeros.m = wp.zeros_like(self.m, requires_grad=requires_grad)
        return zeros

    def empty(self, requires_grad: Optional[bool] = None) -> 'MPMGridData':
        clone = MPMGridData()
        clone.v = wp.empty_like(self.v, requires_grad=requires_grad)
        clone.mv = wp.empty_like(self.mv, requires_grad=requires_grad)
        clone.m = wp.empty_like(self.m, requires_grad=requires_grad)
        return clone


@wp.struct
class MPMConstant(object):

    num_grids: int
    dt: float
    bound: int
    clip_bound: float
    gravity: wp.vec3
    dx: float
    inv_dx: float
    eps: float

    body_friction: float
    body_softness: float
    ground_friction: float

    lower_lim: wp.vec3
    upper_lim: wp.vec3
    env_offsets: wp.array(dtype=wp.vec3)
    num_envs: int = 1


class MPMState(State):

    def __init__(self, shape: int, device: Devicelike = None, requires_grad: bool = False) -> None:

        super().__init__(shape, device, requires_grad)

        particle = MPMParticleData()
        particle.init(shape, device, requires_grad)
        self.particle = particle

    def zero_grad(self) -> None:
        self.particle.zero_grad()

    def clear(self) -> None:
        self.particle.clear()

    def clone(self, requires_grad: Optional[bool] = None) -> 'MPMState':
        clone = MPMState(self.shape, self.device, requires_grad)
        clone.particle = self.particle.clone(requires_grad)
        return clone

    def to_torch(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        def convert(x):
            return wp.to_torch(x, requires_grad=False).requires_grad_(x.requires_grad)

        x = convert(self.particle.x)
        v = convert(self.particle.v)
        C = convert(self.particle.C)
        F = convert(self.particle.F)
        stress = convert(self.particle.stress)
        return x, v, C, F, stress

    def to_torch_grad(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        def convert(x):
            return wp.to_torch(x.grad, requires_grad=False) if x.grad is not None else None

        grad_x = convert(self.particle.x)
        grad_v = convert(self.particle.v)
        grad_C = convert(self.particle.C)
        grad_F = convert(self.particle.F)
        grad_stress = convert(self.particle.stress)
        return grad_x, grad_v, grad_C, grad_F, grad_stress

    def from_torch(
            self,
            x: Optional[Tensor] = None,
            v: Optional[Tensor] = None,
            C: Optional[Tensor] = None,
            F: Optional[Tensor] = None,
            stress: Optional[Tensor] = None) -> None:

        def convert(x, dtype):
            array = wp.from_torch(x.contiguous(), dtype, requires_grad=False)
            array.requires_grad = x.requires_grad
            return array

        if x is not None:
            self.particle.x = convert(x, wp.vec3)
        if v is not None:
            self.particle.v = convert(v, wp.vec3)
        if C is not None:
            self.particle.C = convert(C, wp.mat33)
        if F is not None:
            self.particle.F = convert(F, wp.mat33)
        if stress is not None:
            self.particle.stress = convert(stress, wp.mat33)

    def from_torch_grad(
            self,
            grad_x: Optional[Tensor] = None,
            grad_v: Optional[Tensor] = None,
            grad_C: Optional[Tensor] = None,
            grad_F: Optional[Tensor] = None,
            grad_stress: Optional[Tensor] = None) -> None:
        """
        Warp uses array.grad to store gradients, so we need array.grad.assign(tensor.grad)
        instead of array.grad = tensor.grad. I miss c++ so much.
        """

        def convert(a, t):
            a.grad.assign(wp.from_torch(t.contiguous(), a.dtype, requires_grad=False))

        if grad_x is not None:
            convert(self.particle.x, grad_x)
        if grad_v is not None:
            convert(self.particle.v, grad_v)
        if grad_C is not None:
            convert(self.particle.C, grad_C)
        if grad_F is not None:
            convert(self.particle.F, grad_F)
        if grad_stress is not None:
            convert(self.particle.stress, grad_stress)


class MPMModel(Model):

    ConstantType = MPMConstant
    StaticsType = MPMStatics
    StateType = MPMState

    def __init__(self, constant: ConstantType, device: Devicelike = None, requires_grad: bool = False) -> None:
        super().__init__(constant, device)
        self.requires_grad = requires_grad
        self.grid_op = None
        self.grid_op_name = None

        shape = (self.constant.num_envs, self.constant.num_grids, self.constant.num_grids, self.constant.num_grids)
        grid = MPMGridData()
        grid.init(shape, device, requires_grad)
        self.grid = grid

    def forward(
            self,
            statics: MPMStatics,
            state_curr: MPMState,
            state_next: MPMState,
            tape: Optional[Tape] = None) -> None:

        device = self.device
        constant = self.constant
        particle_curr = state_curr.particle
        particle_next = state_next.particle
        grid = self.grid

        num_envs = constant.num_envs
        num_grids = constant.num_grids
        num_particles = particle_curr.x.shape[0]

        grid.clear()
        grid.zero_grad()

        wp.launch(self.p2g, dim=num_particles, inputs=[constant, statics, particle_curr, grid], device=device)
        wp.launch(self.grid_op, dim=[num_envs] + [num_grids] * 3, inputs=[constant, grid], device=device)

        with CondTape(tape, self.requires_grad):
            wp.launch(self.g2p, dim=num_particles, inputs=[constant, statics, particle_curr, particle_next, grid], device=device)

    def backward(self, statics: MPMStatics, state_curr: MPMState, state_next: MPMState, tape: Tape) -> None:

        device = self.device
        constant = self.constant
        particle_curr = state_curr.particle
        grid = self.grid

        num_envs = constant.num_envs
        num_grids = constant.num_grids
        num_particles = particle_curr.x.shape[0]

        grid.clear()
        grid.zero_grad()

        local_tape = Tape()
        with local_tape:
            wp.launch(self.p2g, dim=num_particles, inputs=[constant, statics, particle_curr, grid], device=device)
            wp.launch(self.grid_op, dim=[num_envs] + [num_grids] * 3, inputs=[constant, grid], device=device)

        tape.backward()

        local_tape.backward()

    @staticmethod
    @wp.kernel
    def p2g(
            constant: ConstantType,
            statics: StaticsType,
            particle_curr: MPMParticleData,
            grid: MPMGridData) -> None:

        p = wp.tid()

        env = statics.env_id[p]
        p_mass = statics.vol[p] * statics.rho[p]

        p_x = (particle_curr.x[p] - constant.env_offsets[env]) * constant.inv_dx
        base_x = int(p_x[0] - 0.5)
        base_y = int(p_x[1] - 0.5)
        base_z = int(p_x[2] - 0.5)
        f_x = p_x - wp.vec3(
            float(base_x),
            float(base_y),
            float(base_z))

        # quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx,fx-1,fx-2]
        wa = wp.vec3(1.5) - f_x
        wb = f_x - wp.vec3(1.0)
        wc = f_x - wp.vec3(0.5)

        # wp.mat33(col_vec, col_vec, col_vec)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )

        stress = (-constant.dt * statics.vol[p] * 4.0 * constant.inv_dx * constant.inv_dx) * particle_curr.stress[p]
        affine = stress + p_mass * particle_curr.C[p]

        for i in range(3):
            for j in range(3):
                for k in range(3):

                    offset = wp.vec3(float(i), float(j), float(k))
                    dpos = (offset - f_x) * constant.dx
                    weight = w[0, i] * w[1, j] * w[2, k]
                    mv = weight * (p_mass * particle_curr.v[p] + affine * dpos)
                    m = weight * p_mass

                    wp.atomic_add(grid.mv[env], base_x + i, base_y + j, base_z + k, mv)
                    wp.atomic_add(grid.m[env], base_x + i, base_y + j, base_z + k, m)

    @staticmethod
    @wp.kernel
    def grid_op_freeslip(
            constant: ConstantType,
            grid: MPMGridData) -> None:

        env, px, py, pz = wp.tid()

        v = wp.vec3(0.0)
        if grid.m[env, px, py, pz] > 0.0:
            v = grid.mv[env, px, py, pz] / (grid.m[env, px, py, pz] + constant.eps) + constant.gravity * constant.dt
        else:
            v = constant.gravity * constant.dt

        if px < constant.bound and v[0] < 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        if py < constant.bound and v[1] < 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        if pz < constant.bound and v[2] < 0.0:
            v = wp.vec3(v[0], v[1], 0.0)
        if px > constant.num_grids - constant.bound and v[0] > 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        if py > constant.num_grids - constant.bound and v[1] > 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        if pz > constant.num_grids - constant.bound and v[2] > 0.0:
            v = wp.vec3(v[0], v[1], 0.0)

        grid.v[env, px, py, pz] = v

    @staticmethod
    @wp.kernel
    def grid_op_noslip(
            constant: ConstantType,
            grid: MPMGridData) -> None:

        env, px, py, pz = wp.tid()

        v = wp.vec3(0.0)
        if grid.m[env, px, py, pz] > 0.0:
            v = grid.mv[env, px, py, pz] / (grid.m[env, px, py, pz] + constant.eps) + constant.gravity * constant.dt
        else:
            v = constant.gravity * constant.dt

        if px < constant.bound and v[0] < 0.0:
            v = wp.vec3(0.0)
        if py < constant.bound and v[1] < 0.0:
            v = wp.vec3(0.0)
        if pz < constant.bound and v[2] < 0.0:
            v = wp.vec3(0.0)
        if px > constant.num_grids - constant.bound and v[0] > 0.0:
            v = wp.vec3(0.0)
        if py > constant.num_grids - constant.bound and v[1] > 0.0:
            v = wp.vec3(0.0)
        if pz > constant.num_grids - constant.bound and v[2] > 0.0:
            v = wp.vec3(0.0)

        grid.v[env, px, py, pz] = v

    @staticmethod
    @wp.kernel
    def grid_op_dexdeform(
        constant: MPMConstant,
        grid: MPMGridData,
        shape_X_bs: wp.array(dtype=wp.transform),
        shape_body: wp.array(dtype=int),
        body_geo: ModelShapeGeometry,
        num_shapes_per_env: int,
        body_curr: wp.array(dtype=wp.transform),
        body_next: wp.array(dtype=wp.transform),
    ) -> None:

        env, px, py, pz = wp.tid()

        v = wp.vec3(0.0)
        if (grid.m[env, px, py, pz] > constant.eps):

            # normalization
            v = grid.mv[env, px, py, pz] * (1.0 / grid.m[env, px, py, pz])

            # gravity
            v = v + constant.gravity * constant.dt

            gx = wp.vec3(
                float(px) * constant.dx,
                float(py) * constant.dx,
                float(pz) * constant.dx,
            )

            shape_start = env * num_shapes_per_env
            shape_end = shape_start + num_shapes_per_env

            # rigid body interaction
            v = MPMModel._grid_op_dexdeform_body(
                env,
                v,
                gx,
                constant,
                shape_X_bs,
                shape_body,
                body_geo,
                shape_start,
                shape_end,
                body_curr,
                body_next,
            )

            # TODO: ground friction assumes up_axis='y'
            up_v = v[1]
            hit_ground = py < constant.bound and up_v < 0.0
            if constant.ground_friction > 0.0 and hit_ground:
                v = v * wp.max(1.0 + constant.ground_friction * up_v / (wp.length(v) + 1e-30), 0.0)

            # boundary condition
            if px < constant.bound and v[0] < 0.0:
                v = wp.vec3(0.0, v[1], v[2])
            if py < constant.bound and v[1] < 0.0:
                v = wp.vec3(v[0], 0.0, v[2])
            if pz < constant.bound and v[2] < 0.0:
                v = wp.vec3(v[0], v[1], 0.0)
            if px > constant.num_grids - constant.bound and v[0] > 0.0:
                v = wp.vec3(0.0, v[1], v[2])
            if py > constant.num_grids - constant.bound and v[1] > 0.0:
                v = wp.vec3(v[0], 0.0, v[2])
            if pz > constant.num_grids - constant.bound and v[2] > 0.0:
                v = wp.vec3(v[0], v[1], 0.0)

        grid.v[env, px, py, pz] = v

    @staticmethod
    @wp.func
    def _grid_op_dexdeform_body(
        env: int,
        v_in: wp.vec3,
        gx: wp.vec3,
        constant: MPMConstant,
        shape_X_bs: wp.array(dtype=wp.transform),
        shape_body: wp.array(dtype=int),
        body_geo: ModelShapeGeometry,
        shape_start: int,
        shape_end: int,
        body_curr: wp.array(dtype=wp.transform),
        body_next: wp.array(dtype=wp.transform),
    ) -> wp.vec3:

        X_wo = wp.transform(constant.env_offsets[env], wp.quat_identity())
        X_ow = wp.transform_inverse(X_wo)

        v = wp.vec3f(v_in)
        for shape_index in range(shape_start, shape_end):
            body_index = shape_body[shape_index]

            X_wb = wp.transform_identity()
            X_wb_next = wp.transform_identity()
            if body_index >= 0:
                X_wb = body_curr[body_index]
                X_wb_next = body_next[body_index]

                # transform world frame to ignore env offset
                X_wb = wp.transform_multiply(X_ow, X_wb)
                X_wb_next = wp.transform_multiply(X_ow, X_wb_next)

            X_bs = shape_X_bs[shape_index]

            X_ws = wp.transform_multiply(X_wb, X_bs)
            X_sw = wp.transform_inverse(X_ws)

            # transform particle position to shape local space
            x_local = wp.transform_point(X_sw, gx)

            # geo description
            geo_type = body_geo.type[shape_index]
            geo_scale = body_geo.scale[shape_index]

            # evaluate shape sdf
            d = 1.0e6
            n = wp.vec3()

            if geo_type == wp.sim.GEO_SPHERE:
                d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
                n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

            if geo_type == wp.sim.GEO_BOX:
                d = box_sdf(geo_scale, x_local)
                n = box_sdf_grad(geo_scale, x_local)

            if geo_type == wp.sim.GEO_CAPSULE:
                d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
                n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

            if geo_type == wp.sim.GEO_CYLINDER:
                d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
                n = cylinder_sdf_grad(geo_scale[0], geo_scale[1], x_local)

            if geo_type == wp.sim.GEO_CONE:
                d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
                n = cone_sdf_grad(geo_scale[0], geo_scale[1], x_local)

            # TODO: geo_type == wp.sim.GEO_MESH

            # TODO: geo_type == wp.sim.GEO_SDF

            if geo_type == wp.sim.GEO_PLANE:
                d = plane_sdf(geo_scale[0], geo_scale[1], x_local)
                n = wp.vec3(0.0, 1.0, 0.0)

            n = wp.normalize(n)

            # contact point in body local space
            body_pos = wp.transform_point(X_bs, x_local - n * d)

            # body position in world space
            bx = wp.transform_point(X_wb, body_pos)

            normal = wp.transform_vector(X_ws, n)
            dist = wp.dot(normal, gx - bx)

            friction = constant.body_friction
            softness = constant.body_softness
            influence = wp.min(wp.exp(-dist * softness), 1.)

            if (influence > 0.1) or (dist <= 0):
                bv = (wp.transform_point(X_wb_next, wp.transform_point(X_bs, x_local)) - gx) / constant.dt
                rel_v = v - bv

                normal_component = wp.dot(rel_v, normal)
                grid_v_t = rel_v - wp.min(normal_component, 0.) * normal

                if (normal_component < 0) and (wp.dot(grid_v_t, grid_v_t) > 1e-30):
                    # apply friction
                    grid_v_t_norm = wp.length(grid_v_t)
                    grid_v_t = grid_v_t * (1. / grid_v_t_norm) * wp.max(0., grid_v_t_norm + normal_component * friction)
                v = bv + rel_v * (1. - influence) + grid_v_t * influence
        return v

    @staticmethod
    @wp.kernel
    def g2p(
            constant: ConstantType,
            statics: StaticsType,
            particle_curr: MPMParticleData,
            particle_next: MPMParticleData,
            grid: MPMGridData) -> None:

        p = wp.tid()

        env = statics.env_id[p]
        p_x = (particle_curr.x[p] - constant.env_offsets[env]) * constant.inv_dx
        base_x = int(p_x[0] - 0.5)
        base_y = int(p_x[1] - 0.5)
        base_z = int(p_x[2] - 0.5)
        f_x = p_x - wp.vec3(
            float(base_x),
            float(base_y),
            float(base_z))

        # quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx,fx-1,fx-2]
        wa = wp.vec3(1.5) - f_x
        wb = f_x - wp.vec3(1.0)
        wc = f_x - wp.vec3(0.5)

        # wp.mat33(col_vec, col_vec, col_vec)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )

        new_v = wp.vec3(0.0)
        new_C = wp.mat33(0.0)

        for i in range(3):
            for j in range(3):
                for k in range(3):

                    offset = wp.vec3(float(i), float(j), float(k))
                    dpos = (offset - f_x) * constant.dx
                    weight = w[0, i] * w[1, j] * w[2, k]

                    v = grid.v[env, base_x + i, base_y + j, base_z + k]
                    new_v = new_v + weight * v
                    new_C = new_C + (4.0 * weight * constant.inv_dx * constant.inv_dx) * wp.outer(v, dpos)

        particle_next.v[p] = new_v
        particle_next.C[p] = new_C
        particle_next.F_trial[p] = (wp.identity(n=3, dtype=float) + constant.dt * new_C) * particle_curr.F[p]

        bound = constant.clip_bound * constant.dx
        new_x = particle_curr.x[p] + constant.dt * new_v
        lo = constant.lower_lim + constant.env_offsets[env]
        up = constant.upper_lim + constant.env_offsets[env]
        new_x = wp.vec3(
            wp.clamp(new_x[0], lo[0] + bound, up[0] - bound),
            wp.clamp(new_x[1], lo[1] + bound, up[1] - bound),
            wp.clamp(new_x[2], lo[2] + bound, up[2] - bound),
        )
        particle_next.x[p] = new_x

    @staticmethod
    @wp.kernel
    def eval_stress(
        constant: MPMConstant,
        statics: MPMStatics,
        particle: MPMParticleData,
    ):
        p = wp.tid()

        m = statics.material_id[p]
        material = statics.material[m]

        if material.name == materials.MATL_PLBPLASTICINE:  # PlasticineLab
            particle.F[p] = materials.plasticine_deformation(particle.F_trial[p], material)
            particle.stress[p] = materials.corotated_elasticity(particle.F[p], material)

        if material.name == materials.MATL_PLASTICINE:
            particle.F[p] = materials.plasticine_deformation(particle.F_trial[p], material)
            particle.stress[p] = materials.sigma_elasticity(particle.F[p], material)

        if material.name == materials.MATL_WATER:
            particle.F[p] = materials.water_deformation(particle.F_trial[p], material)
            particle.stress[p] = materials.volume_elasticity(particle.F[p], material)

        if material.name == materials.MATL_SAND:
            particle.F[p] = materials.sand_deformation(particle.F_trial[p], material)
            particle.stress[p] = materials.sigma_elasticity(particle.F[p], material)

        if material.name == materials.MATL_NEOHOOKEAN:
            particle.F[p] = materials.identity_deformation(particle.F_trial[p], material)
            particle.stress[p] = materials.neohookean_elasticity(particle.F[p], material)

        if material.name == materials.MATL_COROTATED:
            particle.F[p] = materials.identity_deformation(particle.F_trial[p], material)
            particle.stress[p] = materials.corotated_elasticity(particle.F[p], material)


class MPMModelBuilder(ModelBuilder):

    StateType = MPMState
    ConstantType = MPMConstant
    ModelType = MPMModel

    def parse_cfg(
        self,
        cfg: config.physics.sim.BaseSimConfig,
        num_envs: int = 1,
        env_offsets: list | np.ndarray = None,
    ) -> 'MPMModelBuilder':

        num_grids: int = cfg['num_grids']
        dt: float = cfg['dt']
        bound: int = cfg['bound']  # grid bound
        clip_bound: float = cfg['clip_bound']
        gravity: np.ndarray = np.array(cfg['gravity'], dtype=np.float32)
        bc: str = cfg['bc']  # boundary condition
        eps: float = cfg['eps']

        dx: float = 1 / num_grids
        inv_dx: float = float(num_grids)

        lower_lim: np.array = np.array(cfg['lower_lim'], dtype=np.float32)
        upper_lim: np.array = np.array(cfg['upper_lim'], dtype=np.float32)

        body_friction: float = cfg['body_friction']
        body_softness: float = cfg['body_softness']
        ground_friction: float = cfg['ground_friction']

        self.config['num_grids'] = num_grids
        self.config['dt'] = dt
        self.config['bound'] = bound
        self.config['clip_bound'] = clip_bound
        self.config['gravity'] = gravity
        self.config['dx'] = dx
        self.config['inv_dx'] = inv_dx
        self.config['bc'] = bc
        self.config['eps'] = eps

        self.config['lower_lim'] = lower_lim
        self.config['upper_lim'] = upper_lim
        self.config['env_offsets'] = env_offsets
        self.config['num_envs'] = num_envs

        self.config['body_friction'] = body_friction
        self.config['body_softness'] = body_softness
        self.config['ground_friction'] = ground_friction

        return self

    def build_constant(self) -> ConstantType:

        constant = super().build_constant()
        constant.num_grids = self.config['num_grids']
        constant.dt = self.config['dt']
        constant.bound = self.config['bound']
        constant.clip_bound = self.config['clip_bound']
        constant.gravity = wp.vec3(*self.config['gravity'])
        constant.dx = self.config['dx']
        constant.inv_dx = self.config['inv_dx']
        constant.eps = self.config['eps']

        constant.lower_lim = wp.vec3(*self.config['lower_lim'])
        constant.upper_lim = wp.vec3(self.config['upper_lim'])
        constant.env_offsets = wp.array(self.config['env_offsets'], dtype=wp.vec3)
        constant.num_envs = self.config['num_envs']

        constant.body_friction = self.config['body_friction']
        constant.body_softness = self.config['body_softness']
        constant.ground_friction = self.config['ground_friction']

        return constant

    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> ModelType:
        model = super().finalize(device, requires_grad)
        model.grid_op_name = self.config['bc']
        if model.grid_op_name == 'freeslip':
            model.grid_op = MPMModel.grid_op_freeslip
        elif model.grid_op_name == 'noslip':
            model.grid_op = MPMModel.grid_op_noslip
        elif model.grid_op_name == 'dexdeform':
            model.grid_op = MPMModel.grid_op_dexdeform
        else:
            raise ValueError('invalid boundary condition: {}'.format(self.config['bc']))
        return model


@dataclass
class MPMInitData(object):

    rho: float
    num_particles: int
    vol: float
    material: materials.MPMMaterial

    pos: np.ndarray
    lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    center: np.ndarray = None
    ind_vel: np.ndarray = None
    env_id: int = 0

    asset_root = Path(__file__).resolve().parent.parent.parent.parent / 'assets' / 'warp_mpm_sga'

    def __post_init__(self) -> None:
        if self.center is None:
            self.center = self.pos.mean(0)

    @classmethod
    def get(cls, cfg: config.physics.env.BaseEnvConfig) -> 'MPMInitData':
        kwargs: dict = None
        if cfg['shape']['name'].startswith('cube'):
            kwargs = cls.get_cube(
                cfg['shape']['center'],
                cfg['shape']['size'],
                cfg['shape']['resolution'],
                cfg['shape']['mode'],
            )
        elif cfg['shape']['name'].startswith('cylinder'):
            kwargs = cls.get_cylinder(
                cfg['shape']['center'],
                cfg['shape']['size'],
                cfg['shape']['num_particles'],
                cfg['shape']['vol'],
                cfg['shape']['mode'],
            )
        elif cfg['shape']['name'].startswith('mesh'):
            kwargs = cls.get_mesh(
                cfg['shape']['name'],
                cfg['shape']['filepath'],
                cfg['shape']['center'],
                cfg['shape']['size'],
                cfg['shape']['resolution'],
                cfg['shape']['mode'],
                cfg['shape'].get('sort', None),
            )
        else:
            raise ValueError('invalid shape type: {}'.format(cfg['shape']['type']))

        E = cfg['physics'].get('E', cfg['physics'].get('youngs_modulus', None))
        nu = cfg['physics'].get('nu', cfg['physics'].get('poissons_ratio', None))
        yield_stress = cfg['physics'].get('yield_stress', None)
        cohesion = cfg['physics'].get('cohesion', None)
        alpha = cfg['physics'].get('alpha', None)
        material = materials.get_material(
            cfg['physics']['material'],
            E=E,
            nu=nu,
            yield_stress=yield_stress,
            cohesion=cohesion,
            alpha=alpha,
        )

        return cls(rho=cfg['rho'], material=material, **kwargs)

    @classmethod
    def get_mesh(
            cls,
            name: str,
            filepath: str,
            center: list | np.ndarray,
            size: list | np.ndarray,
            resolution: int,
            mode: str,
            sort: Optional[int]) -> Mapping[str, Any]:

        center = np.array(center)
        size = np.array(size)

        fn = filepath.split('/')[-1].split('.')[0]

        asset_root = cls.asset_root
        precompute_name = f'{name}_{fn}_{resolution}_{mode}.npz'

        if (asset_root / precompute_name).is_file():
            file = np.load(asset_root / precompute_name)
            p_x = file['p_x']
            vol = file['vol']
        else:

            import trimesh

            mesh: trimesh.Trimesh = trimesh.load(asset_root / '..' / f'{filepath}', force='mesh')

            # if not mesh.is_watertight:
            #     raise ValueError('invalid mesh: not watertight')

            bounds = mesh.bounds.copy()

            if mode == 'uniform':

                mesh.vertices = (mesh.vertices - bounds[0]) / (bounds[1] - bounds[0])
                dims = np.linspace(np.zeros(3), np.ones(3), resolution).T
                grid = np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1).reshape(-1, 3)
                p_x = grid[mesh.contains(grid)]
                p_x = (p_x * (bounds[1] - bounds[0]) + bounds[0] - bounds.mean(0)) / (bounds[1] - bounds[0]).max()

            else:
                raise ValueError('invalid mode: {}'.format(mode))

            if sort is not None:
                indices = np.array(list(sorted(range(p_x.shape[0]), reverse=True, key=lambda x: p_x[:, sort][x])))
                p_x = p_x[indices]

            vol = mesh.volume / p_x.shape[0]
            np.savez(asset_root / precompute_name, p_x=p_x, vol=vol)

        vol = vol * np.prod(size)
        p_x = p_x * size + center
        p_x = np.ascontiguousarray(p_x.reshape(-1, 3))

        return {'num_particles': p_x.shape[0], 'vol': vol, 'pos': p_x, 'center': center}

    @classmethod
    def get_cube(
            cls,
            center: list | np.ndarray,
            size: list | np.ndarray,
            resolution: int,
            mode: str) -> dict[str, Any]:

        asset_root = cls.asset_root
        size_str = '_'.join([str(s) for s in size])
        precompute_name = f'cube_{resolution}_{mode}_{size_str}.npz'

        center = np.array(center)
        size = np.array(size)

        if (asset_root / precompute_name).is_file():
            file = np.load(asset_root / precompute_name)
            p_x = file['p_x']
            vol = file['vol']
        else:
            resolutions = np.around(size * resolution / size.max()).astype(int)
            if mode == 'uniform':
                dims = [np.linspace(l, r, res) for l, r, res in zip(-size / 2, size / 2, resolutions)]
                p_x = np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1).reshape(-1, 3)
            elif mode == 'random':
                rng = np.random.Generator(np.random.PCG64(0))
                p_x = rng.uniform(-size / 2, size / 2, (np.prod(resolutions), 3)).reshape(-1, 3)
            else:
                raise ValueError('invalid mode: {}'.format(mode))

            vol = np.prod(size) / p_x.shape[0]
            np.savez(asset_root / precompute_name, p_x=p_x, vol=vol)

        p_x = p_x + center
        p_x = np.ascontiguousarray(p_x.reshape(-1, 3))
        return {'num_particles': p_x.shape[0], 'vol': vol, 'pos': p_x, 'center': center}

    @classmethod
    def get_cylinder(
            cls,
            center: list | np.ndarray,
            size: list | np.ndarray,
            num_particles: int,
            vol: float,
            mode: str) -> dict[str, Any]:

        asset_root = cls.asset_root
        size_str = '_'.join([str(s) for s in size])
        precompute_name = f'cylinder_{num_particles}_{mode}_{size_str}.npz'

        center = np.array(center)
        size = np.array(size)
        h, r, _ = size

        if (asset_root / precompute_name).is_file():
            file = np.load(asset_root / precompute_name)
            p_x = file['p_x']
            assert vol == file['vol']
        else:
            if mode == 'uniform':
                raise NotImplementedError
            elif mode == 'random':
                rng = np.random.Generator(np.random.PCG64(0))

                sdf_func = shapes.compute_cylinder_sdf(h=h, r=r)
                sample_func = shapes.box_particles(np.array([r, h, r]))

                p_x = shapes.rejection_sampling(
                    init_pos=np.array([0., 0., 0.]),
                    n_particles=num_particles,
                    sample_func=sample_func,
                    sdf_func=sdf_func,
                    rng=rng,
                )
            elif mode == 'random_symmetric':
                rng = np.random.Generator(np.random.PCG64(0))

                sdf_func = shapes.compute_cylinder_sdf(h=h, r=r, half=True)
                sample_func = shapes.box_particles(np.array([r, h, r]))

                p_x = shapes.rejection_sampling(
                    init_pos=np.array([0., 0., 0.]),
                    n_particles=num_particles // 2,
                    sample_func=sample_func,
                    sdf_func=sdf_func,
                    rng=rng,
                )

                p_x_mirror = p_x * np.array([-1, 1, 1])
                p_x = np.stack([p_x, p_x_mirror], axis=0)
            else:
                raise ValueError('invalid mode: {}'.format(mode))

            # TODO: compute volume if not provided

            np.savez(asset_root / precompute_name, p_x=p_x, vol=vol)

        p_x = p_x + center
        p_x = np.ascontiguousarray(p_x.reshape(-1, 3))
        return {'num_particles': p_x.shape[0], 'vol': vol, 'pos': p_x, 'center': center}

    def set_env_id(self, env_id: int) -> None:
        self.env_id = env_id

    def set_center(self, value: list | np.ndarray) -> None:
        self.pos = self.pos - self.center + np.array(value)
        self.center = np.array(value)

    def set_lin_vel(self, value: list | np.ndarray) -> None:
        self.lin_vel = np.array(value)

    def zero_lin_vel(self) -> None:
        self.set_lin_vel(np.zeros_like(self.lin_vel))

    def set_ang_vel(self, value: list | np.ndarray) -> None:
        self.ang_vel = np.array(value)

    def zero_ang_vel(self) -> None:
        self.set_ang_vel(np.zeros_like(self.ang_vel))

    def set_ind_vel(self, ind_vel: np.ndarray) -> None:
        self.ind_vel = np.array(ind_vel)


class MPMStateInitializer(StateInitializer):

    StateType = MPMState
    ModelType = MPMModel

    def __init__(self, model: ModelType) -> None:
        super().__init__(model)
        self.groups: list[MPMInitData] = []

    def add_group(self, group: MPMInitData) -> None:
        self.groups.append(group)

    def finalize(self, requires_grad: bool = False) -> tuple[StateType, list[int]]:

        pos_groups = []
        vel_groups = []
        sections = []

        for group in self.groups:
            pos = group.pos.copy()

            if group.ind_vel is None:
                lin_vel = group.lin_vel.copy()
                ang_vel = group.ang_vel.copy()
                vel = lin_vel + np.cross(ang_vel, pos - group.center)
            else:
                vel = group.ind_vel.copy()

            pos_groups.append(pos)
            vel_groups.append(vel)
            sections.append(group.num_particles)

        pos_groups = np.concatenate(pos_groups, axis=0)
        vel_groups = np.concatenate(vel_groups, axis=0)

        state_0 = super().finalize(shape=pos_groups.shape[0], requires_grad=requires_grad)

        pos_groups = pos_groups.astype(dtype=np.float32)
        vel_groups = vel_groups.astype(dtype=np.float32)

        state_0.particle.x.assign(wp.from_numpy(pos_groups))
        state_0.particle.v.assign(wp.from_numpy(vel_groups))

        return state_0, sections


class MPMStaticsInitializer(StaticsInitializer):

    StaticsType = MPMStatics
    ModelType = MPMModel

    def __init__(self, model: ModelType) -> None:
        super().__init__(model)
        self.groups: list[MPMInitData] = []

        self.sections: list[int] = []
        self.vols: list[float] = []
        self.rhos: list[float] = []

        self.env_ids: list[float] = []
        self.material_ids: list[int] = []
        self.materials: list[materials.MPMMaterial] = []

    def add_group(self, group: MPMInitData) -> None:
        self.groups.append(group)

    def finalize(self) -> StaticsType:

        for i, group in enumerate(self.groups):
            self.sections.append(group.num_particles)
            self.vols.append(group.vol)
            self.rhos.append(group.rho)

            self.env_ids.append(group.env_id)
            self.material_ids.append(i)
            self.materials.append(group.material)

        statics = super().finalize(shape=sum(self.sections))
        statics.update_vol(self.sections, self.vols)
        statics.update_rho(self.sections, self.rhos)
        statics.update_env_id(self.sections, self.env_ids)
        statics.update_material(self.sections, self.material_ids, self.materials)

        return statics
