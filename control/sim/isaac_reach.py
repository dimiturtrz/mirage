"""hxq.8 — a PhysX rigid-body reach env satisfying `core.rollout.Env`, the fidelity rung above the numpy
point mass.

Same task, same observation, same governing equation — a real solver instead of explicit Euler. A dynamic
sphere floats in a zero-gravity scene; each step a planar force `F = gain·clip(a)` is applied and PhysX
integrates one `dt`. With PhysX linear damping set to `drag/mass`, the equation of motion is

    dv/dt = F/mass − (drag/mass)·v = (gain·clip(a) − drag·v) / mass

— exactly `control.point_mass.PointMassReach`, now integrated by PhysX (substeps + implicit damping) rather
than a hand-rolled Euler step. **sim vs real still differ only in `Phys`** (payload `mass`, actuator `gain`),
so the identical sim-to-real policy gap is re-measured at higher fidelity **without touching the policy, the
rollout spine, or the metric** — only this new `Env` implementation. The `omni`/`isaacsim` imports are local
to `boot` so this module imports cleanly before the kit app exists; the runner (`isaac_gap.py`) boots the
`SimulationApp` first, then calls `IsaacReach.boot`.

Not CI-runnable (GPU + a ~30 s kit boot, needs the `sim` extra) — an opt-in fidelity check, run manually.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float

from control.point_mass import Phys, Task
from core.rollout import StepResult

if TYPE_CHECKING:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicSphere
    from isaacsim.core.prims import RigidPrim
    from pxr import PhysxSchema

_DIM = 2                 # planar reach (x, y); z is held constant by zero gravity + planar forcing
_Z = 0.5                 # float height above the origin (no ground contact; gravity is off anyway)
_BALL = "/World/ball"


class IsaacReach:
    """The point-mass reach task on a PhysX rigid sphere — a `core.rollout.Env` (`reset`/`step`).

    One sphere is reused across episodes (teleported home on `reset`, not re-spawned). `configure` sets the
    per-episode seed + the `Phys` (mass/gain), so a factory closure yields matched goals across sim/real."""

    def __init__(self, world: World, sphere: DynamicSphere, view: RigidPrim,  # noqa: PLR0913
                 physx_rb: PhysxSchema.PhysxRigidBodyAPI, task: Task, dt: float):
        self._world = world
        self._sphere = sphere
        self._view = view
        self._physx_rb = physx_rb          # PhysxRigidBodyAPI on the sphere prim (live linear-damping knob)
        self._task = task
        self._rng = np.random.default_rng(0)
        self._mass = 1.0
        self._gain = 1.0
        self._drag = 0.1
        self._amax = 1.0
        self._dt = dt
        self._goal = np.zeros(_DIM)
        self._t = 0

    @classmethod
    def boot(cls, task: Task, phys_dt: float) -> IsaacReach:
        """Build the world + sphere AFTER the SimulationApp is up (local kit imports). Zero gravity so the
        body stays planar and the only forces are the actuator's."""
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicSphere
        from isaacsim.core.prims import RigidPrim
        from pxr import PhysxSchema

        world = World(stage_units_in_meters=1.0, physics_dt=phys_dt, rendering_dt=phys_dt)
        world.get_physics_context().set_gravity(0.0)
        sphere = DynamicSphere(prim_path=_BALL, position=[0.0, 0.0, _Z], radius=0.05, mass=1.0)
        world.scene.add(sphere)
        world.reset()
        view = RigidPrim(prim_paths_expr=_BALL)
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(sphere.prim)
        physx_rb.CreateLinearDampingAttr(0.0)
        physx_rb.CreateAngularDampingAttr(0.0)
        return cls(world, sphere, view, physx_rb, task, phys_dt)

    def configure(self, seed: int, phys: Phys) -> IsaacReach:
        """Point the reused env at one episode's seed + physics (payload mass, actuator gain)."""
        self._rng = np.random.default_rng(seed)
        self._mass, self._gain, self._drag, self._amax = phys.mass, phys.gain, phys.drag, phys.amax
        return self

    def _observe(self) -> Float[np.ndarray, "4"]:
        pos, _ = self._sphere.get_world_pose()
        vel = self._sphere.get_linear_velocity()
        return np.concatenate([self._goal - np.asarray(pos)[:_DIM], np.asarray(vel)[:_DIM]]).astype(float)

    def reset(self) -> Float[np.ndarray, "4"]:
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        self._goal = self._task.goal_radius * np.array([np.cos(angle), np.sin(angle)])
        self._sphere.set_world_pose(position=np.array([0.0, 0.0, _Z]))
        self._sphere.set_linear_velocity(np.zeros(3))
        self._sphere.set_mass(self._mass)
        self._physx_rb.GetLinearDampingAttr().Set(self._drag / self._mass)   # dv/dt gains −(drag/mass)·v
        self._t = 0
        return self._observe()

    def step(self, action: Float[np.ndarray, "2"]) -> StepResult:
        a = np.clip(np.asarray(action, dtype=float), -self._amax, self._amax)
        force = np.array([[self._gain * a[0], self._gain * a[1], 0.0]], dtype=np.float32)
        self._view.apply_forces(force, is_global=True)
        self._world.step(render=False)
        self._t += 1
        pos, _ = self._sphere.get_world_pose()
        dist = float(np.linalg.norm(np.asarray(pos)[:_DIM] - self._goal))
        success = dist < self._task.eps
        done = success or self._t >= self._task.horizon
        return StepResult(obs=self._observe(), reward=-dist * self._dt, done=done, success=success)
