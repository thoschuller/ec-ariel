"""(A-lamo): description of script.

Pyramid on top of flat terrain.

Date:       2025-07-23
Status:     Completed ✅
"""


import mujoco
import numpy as np
from src.ariel.utils.mjspec_ops import compute_geom_bounding_box

USE_DEGREES = False


class StairsWorld:
    """MuJoCo world with stairs placed on a flat terrain."""

    def __init__(
        self,
        grid_loc: list[float] = [0, 0],
        num_stairs: int = 4,
        direction: int = 1,
        name: str = "stair",
        floor_size: tuple[float, float, float] = (10, 10, 0.05),
    ) -> None:
        self.SQUARE_LENGTH = 2
        self.V_SIZE = 0.076
        self.H_SIZE = 0.12
        self.H_STEP = self.H_SIZE * 2
        self.V_STEP = self.V_SIZE * 2
        self.BROWN = [0.460, 0.362, 0.216, 1.0]

        self.grid_loc = grid_loc
        self.num_stairs = num_stairs
        self.direction = direction
        self.name = name
        self.floor_size = floor_size

        # --- Root ---
        spec = mujoco.MjSpec()
        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.autolimits = True
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False
        spec.visual.global_.offheight = 960
        spec.visual.global_.offwidth = 1280

        # Set default geom type to BOX
        spec.default.geom.type = mujoco.mjtGeom.mjGEOM_BOX

        # --- Terrain ---
        grid_name = "grid"
        spec.add_texture(
            name=grid_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=600,
            height=600,
        )
        spec.add_material(
            name=grid_name,
            textures=["", f"{grid_name}"],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0.2,
        )
        spec.worldbody.add_light(name="light", pos=[0, 0, 1], castshadow=False)
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            material=grid_name,
            size=self.floor_size,
        )

        # --- Stairs ---
        self._build_stairs(spec)

        # Save the spec
        self.spec: mujoco.MjSpec = spec

    def _build_stairs(self, spec: mujoco.MjSpec) -> None:
        """Build stair geometry above flat floor."""
        z_offset = self.floor_size[2]  # Height of the floor
        body = spec.worldbody.add_body(pos=self.grid_loc + [0], name=self.name)

        x_beginning = -self.SQUARE_LENGTH + self.H_SIZE
        x_end = self.SQUARE_LENGTH - self.H_SIZE
        y_end = -self.SQUARE_LENGTH + self.H_SIZE
        y_beginning = self.SQUARE_LENGTH - self.H_SIZE

        size_one = [self.H_SIZE, self.SQUARE_LENGTH, self.V_SIZE]
        size_two = [self.SQUARE_LENGTH, self.H_SIZE, self.V_SIZE]

        x_pos_l = [x_beginning, 0, self.direction * self.V_SIZE]
        x_pos_r = [x_end, 0, self.direction * self.V_SIZE]
        y_pos_up = [0, y_beginning, self.direction * self.V_SIZE]
        y_pos_down = [0, y_end, self.direction * self.V_SIZE]

        for i in range(self.num_stairs):
            size_one[1] = self.SQUARE_LENGTH - self.H_STEP * i
            size_two[0] = self.SQUARE_LENGTH - self.H_STEP * i

            stair_z = self.direction * (self.V_SIZE + self.V_STEP * i)
            x_pos_l[2] = x_pos_r[2] = y_pos_up[2] = y_pos_down[2] = stair_z

            x_pos_l[0] = x_beginning + self.H_STEP * i
            x_pos_r[0] = x_end - self.H_STEP * i
            y_pos_up[1] = y_beginning - self.H_STEP * i
            y_pos_down[1] = y_end + self.H_STEP * i

            body.add_geom(pos=x_pos_l.copy(), size=size_one.copy(), rgba=self.BROWN)
            body.add_geom(pos=x_pos_r.copy(), size=size_one.copy(), rgba=self.BROWN)
            body.add_geom(pos=y_pos_up.copy(), size=size_two.copy(), rgba=self.BROWN)
            body.add_geom(pos=y_pos_down.copy(), size=size_two.copy(), rgba=self.BROWN)

        top_size = [
            self.SQUARE_LENGTH - self.H_STEP * self.num_stairs,
            self.SQUARE_LENGTH - self.H_STEP * self.num_stairs,
            self.V_SIZE,
        ]
        top_pos = [0, 0, self.direction * (self.V_SIZE + self.V_STEP * self.num_stairs)]
        body.add_geom(pos=top_pos, size=top_size, rgba=self.BROWN)

    def spawn(
        self,
        mj_spec: mujoco.MjSpec,
        spawn_position: list[float, float, float] | None = None,
        *,
        small_gap: float = 0.01,
        correct_for_bounding_box: bool = True,
        beside_direction: str = "left",  # 'left' or 'right'
        offset: float = 5,
    ) -> None:
        """
        Spawn a robot next to the stairs on flat terrain.

        Parameters
        ----------
        mj_spec : mujoco.MjSpec
            The MuJoCo robot spec.
        spawn_position : list[float, float, float] | None, optional
            Optional exact position.
        small_gap : float, optional
            Gap above floor for stability.
        correct_for_bounding_box : bool, optional
            Adjust spawn height by bounding box.
        beside_direction : str, optional
            'left' or 'right' of stairs in X.
        offset : float, optional
            Horizontal distance from center.
        """
        x_offset = -offset if beside_direction == "left" else offset
        z_base = self.floor_size[2]  # Top of the flat floor

        if spawn_position is None:
            spawn_position = [self.grid_loc[0] + x_offset, self.grid_loc[1], z_base]

        if correct_for_bounding_box:
            model = mj_spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data, nstep=10)
            min_corner, _ = compute_geom_bounding_box(model, data)
            spawn_position[2] -= min_corner[2]

        spawn_position[2] += small_gap

        spawn_site = self.spec.worldbody.add_site(pos=np.array(spawn_position))
        spawn = spawn_site.attach_body(body=mj_spec.worldbody, prefix="robot-")
        spawn.add_freejoint()
