import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh

from yuju.config import AutoSaveConfig
from yuju.geometry_utils import (
    CANONICAL_BOX_FACES,
    CANONICAL_BOX_VERTICES,
    compute_pressure_center,
    compute_torque,
    compute_transform_matrix,
    decompose_matrix,
    estimate_densest_point_distance,
    fit_obb_to_points,
    normalize_angle,
)
from yuju.ui_utils import ui_tree_node
from yuju.utils import load_plt_file

logger = logging.getLogger(__name__)

DEFAULT_BOX_PARAM = MappingProxyType({
    "tx": 0.0,
    "ty": 0.0,
    "tz": 0.0,
    "rx": 0.0,
    "ry": 0.0,
    "rz": 0.0,
    "sx": 1.0,
    "sy": 1.0,
    "sz": 1.0,
    "padding": 0.0,
})


class PolyscopeApp:
    def __init__(self, data_path: str | Path, mesh_path: str | Path):
        self.field = load_plt_file(data_path)  # (N, 6)

        try:
            self.config = AutoSaveConfig(".yuju.json")
            logger.debug("Loaded config from .yuju.json")
        except FileNotFoundError:
            logger.debug("Using default config.")
            self.config = AutoSaveConfig(".yuju.json")

        logger.debug(f"Loading mesh from {mesh_path}")
        self.input_mesh = trimesh.load_mesh(mesh_path)

        ps.set_program_name("Yuju")
        ps.set_print_prefix("[Yuju][Polyscope] ")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("z_up")
        ps.set_front_dir("x_front")

        ps.init()

        self._enable_vec_visualization = False

        self._densest_point_distance = estimate_densest_point_distance(
            self.field[:, :3],
            k=min(5000, self.field.shape[0]),
            quantile=0.01,
        )
        logger.debug(f"Densest point distance: {self._densest_point_distance:.3f}")

        self._radius = 0.1 * self._densest_point_distance

        self.ps_f_pc = ps.register_point_cloud("input_points", self.field[:, :3], color=(0.2, 0.5, 0.5), enabled=True)
        self.ps_f_pc.set_radius(self._radius, relative=False)

        self.ps_f_vec = self.ps_f_pc.add_vector_quantity(
            "input_vecs",
            self.unit_vec * self._radius * 5,
            radius=self._radius * 0.025,
            vectortype="ambient",
            color=(0.2, 0.5, 0.5),
            enabled=self._enable_vec_visualization,
        )

        self.ps_mesh = ps.register_surface_mesh(
            "input_mesh", self.input_mesh.vertices, self.input_mesh.faces, color=(0.7, 0.7, 0.9)
        )

        self.picked_points: list[np.ndarray] = []
        self.picked_cloud = None

        self.box_mesh = ps.register_surface_mesh(
            "box", CANONICAL_BOX_VERTICES, CANONICAL_BOX_FACES, color=(1.0, 0.0, 0.0), transparency=0.4
        )
        self._update_box_geometry()

    @property
    def pos(self):
        return self.field[:, :3]

    @property
    def vec(self):
        return self.field[:, 3:6]

    @property
    def unit_vec(self):
        return self.vec / np.linalg.norm(self.vec, axis=1, keepdims=True)

    @property
    def box_params(self) -> dict[str, float]:
        keys = ["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz", "padding"]
        for k in keys:
            if k not in self.config:
                self.config[k] = DEFAULT_BOX_PARAM[k]
        return {k: self.config[k] for k in keys}

    @property
    def reference_point(self) -> np.ndarray:
        if "reference_point" not in self.config:
            self.config["reference_point"] = [0.0, 0.0, 0.0]
        return np.array(self.config["reference_point"])

    @reference_point.setter
    def reference_point(self, value: np.ndarray):
        self.config["reference_point"] = value.tolist()

    def update_box_params(self, params: dict[str, float]):
        for k, v in params.items():
            self.config[k] = float(v)

    def _update_box_geometry(self):
        transform = compute_transform_matrix(**self.box_params)
        self.box_mesh.set_transform(transform)

    def _points_in_box(self) -> np.ndarray:
        transform = self.box_mesh.get_transform()
        inv_transform = np.linalg.inv(transform)
        homog_pos = np.hstack([self.pos, np.ones((self.pos.shape[0], 1))])
        local_pos = (inv_transform @ homog_pos.T).T[:, :3]
        return np.all((local_pos >= -0.5) & (local_pos <= 0.5), axis=1)

    def _box_center_and_diagonal(self) -> tuple[np.ndarray, float]:
        transform = self.box_mesh.get_transform()
        center_local = np.array([0.0, 0.0, 0.0, 1.0])
        center_world = (transform @ center_local)[:3]
        verts_h = np.hstack([CANONICAL_BOX_VERTICES, np.ones((8, 1))])
        verts_world = (transform @ verts_h.T).T[:, :3]
        min_corner = verts_world.min(axis=0)
        max_corner = verts_world.max(axis=0)
        diag_len = np.linalg.norm(max_corner - min_corner)
        return center_world, 0.5 * diag_len  # type: ignore[return-value]

    def _ui_reference_point_inputs(self):
        with ui_tree_node("Torque Reference Point") as expanded:
            if not expanded:
                return
            changed_x, val_x = psim.InputFloat("Ref X", self.reference_point[0])
            changed_y, val_y = psim.InputFloat("Ref Y", self.reference_point[1])
            changed_z, val_z = psim.InputFloat("Ref Z", self.reference_point[2])
            if changed_x or changed_y or changed_z:
                self.reference_point = np.array([val_x, val_y, val_z], dtype=float)
        psim.Separator()

    def _ui_compute_button(self):
        if psim.Button("Compute"):
            inside_mask = self._points_in_box()
            if not np.any(inside_mask):
                logger.info("No points inside the box.")
                return

            pos_sel = self.pos[inside_mask]
            vec_sel = self.vec[inside_mask]

            force_vec = vec_sel.sum(axis=0)
            force_norm = np.linalg.norm(force_vec)

            pressure_center = compute_pressure_center(pos_sel, vec_sel)
            torque_vec, torque_norm = compute_torque(pos_sel, vec_sel, self.reference_point)

            print("Results:")
            print(f"  Pressure center: \t{pressure_center}")
            print(f"  Force vector: \t{force_vec} \tNorm: {force_norm}")
            print(f"  Torque vector (wrt {self.reference_point}): \t{torque_vec} \tNorm: {torque_norm}")

            _, half_diag = self._box_center_and_diagonal()
            dir_normed = force_vec / force_norm if force_norm > 1e-8 else np.zeros(3)
            arrow_vec = dir_normed * half_diag

            ps.remove_point_cloud("pressure_center_point", error_if_absent=False)
            pt = ps.register_point_cloud(
                "pressure_center_point", pressure_center.reshape(1, 3), color=(1.0, 0.5, 0.0), radius=0.03
            )
            pt.add_vector_quantity(
                "force_direction",
                arrow_vec.reshape(1, 3),
                vectortype="ambient",
                color=(0.0, 0.0, 1.0),
                radius=0.015,
                enabled=True,
            )

            base_color = np.tile(np.array([[0.2, 0.5, 0.5]]), (self.pos.shape[0], 1))
            base_color[inside_mask] = np.array([1.0, 0.0, 0.0])
            self.ps_f_pc.add_color_quantity("inside_highlight", base_color, enabled=True)

    def _fit_bbox_to_points_and_update_params(self, points: np.ndarray) -> bool:
        try:
            fitted_params = fit_obb_to_points(points, padding=self.box_params["padding"])
            self.update_box_params(fitted_params)
        except ValueError as e:
            if "At least 4 points are required to fit an OBB" not in str(e):
                logger.warning(f"Could not fit bounding box: {e}")
            return False
        except Exception:
            logger.exception("Unexpected error during OBB fitting.")
            return False
        else:
            return True

    def _handle_mouse_picking(self, io: Any):
        if io.MouseClicked[0] and io.KeyShift:
            pick_res = ps.pick(screen_coords=io.MousePos)
            if pick_res.is_hit and pick_res.structure_name == "picked_points":
                if self.picked_points:
                    idx = pick_res.local_index
                    if 0 <= idx < len(self.picked_points):
                        self.picked_points.pop(idx)
            elif pick_res.is_hit and pick_res.structure_name == "input_mesh":
                self.picked_points.append(pick_res.position.copy())
            if self.picked_cloud is not None:
                ps.remove_point_cloud("picked_points", error_if_absent=False)
                self.picked_cloud = None
            if self.picked_points:
                pts_np = np.array(self.picked_points)
                self.picked_cloud = ps.register_point_cloud("picked_points", pts_np, color=(0.2, 1.0, 0.2), radius=0.01)
                if self._fit_bbox_to_points_and_update_params(pts_np):
                    self._update_box_geometry()

    def _handle_transform_sliders(self):
        for k, s in {
            "tx": 0.01,
            "ty": 0.01,
            "tz": 0.01,
            "rx": 1.0,
            "ry": 1.0,
            "rz": 1.0,
            "sx": 0.01,
            "sy": 0.01,
            "sz": 0.01,
        }.items():
            changed, val = psim.DragFloat(k, self.box_params[k], s, -1000, 1000)
            if changed:
                self.update_box_params({k: normalize_angle(val) if k in ["rx", "ry", "rz"] else val})
                self._update_box_geometry()

    def _handle_padding_slider(self):
        changed, val = psim.DragFloat("padding", self.box_params["padding"], 0.01, 0, 1000)
        if changed:
            self.update_box_params({"padding": max(0.0, val)})
            if self.picked_points:
                pts_np = np.array(self.picked_points)
                if self._fit_bbox_to_points_and_update_params(pts_np):
                    self._update_box_geometry()
            else:
                self._update_box_geometry()

    def _ui_enable_gizmo(self):
        changed, enable_gizmo = psim.Checkbox("Enable Gizmo", self.box_mesh.get_transform_gizmo_enabled())
        if changed:
            self.box_mesh.set_transform_gizmo_enabled(enable_gizmo)

    def _ui_show_points(self):
        changed, show_points = psim.Checkbox(
            "Show Selection Points", self.picked_cloud is not None and self.picked_cloud.is_enabled()
        )
        if changed:
            if self.picked_cloud is None:
                return
            self.picked_cloud.set_enabled(show_points)

    def _ui_reset(self):
        if psim.Button("Reset"):
            self.update_box_params(dict(DEFAULT_BOX_PARAM))
            self._update_box_geometry()
            self.picked_points = []
            if self.picked_cloud:
                ps.remove_point_cloud("picked_points", error_if_absent=False)

    def _ui_field(self):
        need_update_vec = False
        changed, enable_vec_visualization = psim.Checkbox("Show Field", self._enable_vec_visualization)
        if changed:
            self._enable_vec_visualization = enable_vec_visualization
            need_update_vec = True
        v_min = self._densest_point_distance * 0.01
        v_max = self._densest_point_distance * 0.20
        v_speed = (v_max - v_min) / 1000
        radius_changed, radius = psim.DragFloat(
            "Point Radius",
            self._radius,
            v_speed=v_speed,
            v_min=v_min,
            v_max=v_max,
            format="%.4g",
        )
        if radius_changed:
            need_update_vec = True
            self._radius = radius
            self.ps_f_pc.set_radius(radius, relative=False)
        if need_update_vec:
            self.ps_f_pc.remove_quantity("input_vecs", error_if_absent=False)
            self.ps_f_vec = self.ps_f_pc.add_vector_quantity(
                "input_vecs",
                self.unit_vec * self._radius * 5,
                radius=self._radius * 0.025,
                vectortype="ambient",
                color=(0.2, 0.5, 0.5),
                enabled=self._enable_vec_visualization,
            )

    def callback(self) -> None:
        io = psim.GetIO()
        self._handle_mouse_picking(io)
        current_ps_transform = self.box_mesh.get_transform()
        (tx, ty, tz), (rx, ry, rz), (sx_e, sy_e, sz_e) = decompose_matrix(current_ps_transform)
        self.update_box_params({
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "sx": max(0.01, sx_e - 2 * self.box_params["padding"]),
            "sy": max(0.01, sy_e - 2 * self.box_params["padding"]),
            "sz": max(0.01, sz_e - 2 * self.box_params["padding"]),
        })
        with ui_tree_node("Oriented Bounding Box") as expanded:
            if expanded:
                self._handle_transform_sliders()
                self._handle_padding_slider()
        psim.Separator()
        with ui_tree_node("OBB Misc") as expanded:
            if expanded:
                self._ui_enable_gizmo()
                psim.SameLine()
                self._ui_show_points()
                psim.SameLine()
                self._ui_reset()
        with ui_tree_node("Field") as expanded:
            if expanded:
                self._ui_field()
        psim.Separator()
        self._ui_reference_point_inputs()
        self._ui_compute_button()
        self._update_box_geometry()

    def run(self):
        ps.set_user_callback(self.callback)
        ps.show()
