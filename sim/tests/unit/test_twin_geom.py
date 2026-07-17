"""Unit tests for the pxr-free twin geometry helpers (twin_geom) — OBJ parse, depth back-projection,
Gaussian-patch defect. No Isaac boot required: they run in any numpy env."""
import numpy as np

from twin_geom import backproject, deform, parse_obj


class TestParseObj:
    def test_verts_and_faces_zero_indexed(self, tmp_path):
        obj = tmp_path / "m.obj"
        obj.write_text("# comment\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1 2 3\n")
        verts, faces = parse_obj(str(obj))
        assert verts.shape == (3, 3) and verts.dtype == np.float32
        assert faces.shape == (1, 3) and faces.dtype == np.int32
        np.testing.assert_array_equal(faces[0], [0, 1, 2])          # 1-based OBJ -> 0-based
        np.testing.assert_allclose(verts[1], [1, 0, 0])

    def test_face_slash_takes_vertex_index_only(self, tmp_path):
        obj = tmp_path / "m.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/4/7 2/5/8 3/6/9\n")
        _, faces = parse_obj(str(obj))
        np.testing.assert_array_equal(faces[0], [0, 1, 2])          # tex/normal indices dropped


class TestBackproject:
    def test_center_is_origin_edges_scale(self):
        depth = np.full((4, 4), 2.0, np.float32)
        xyz = backproject(depth, fx=2.0, fy=2.0)
        assert xyz.shape == (4, 4, 3)
        np.testing.assert_allclose(xyz[..., 2], 2.0)               # z channel = depth
        np.testing.assert_allclose(xyz[2, 2, :2], [0.0, 0.0])      # pixel (w/2,h/2) -> origin
        np.testing.assert_allclose(xyz[0, 0, :2], [-2.0, -2.0])    # (u-w/2)/fx*z = (0-2)/2*2

    def test_infinite_depth_maps_to_zero(self):
        depth = np.array([[np.inf, 1.0]], np.float32)
        xyz = backproject(depth, fx=1.0, fy=1.0)
        np.testing.assert_array_equal(xyz[0, 0], [0.0, 0.0, 0.0])  # miss (inf) -> 0
        assert xyz[0, 1, 2] == 1.0


class TestDeform:
    def test_only_z_changes_bounded_by_amp(self):
        rng = np.random.RandomState(0)
        xy = np.stack(np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), -1).reshape(-1, 2)
        verts = np.concatenate([xy, np.zeros((len(xy), 1))], 1).astype(np.float32)
        out, amp = deform(verts, rng, sign=1.0)
        assert out.shape == verts.shape and amp > 0
        np.testing.assert_array_equal(out[:, :2], verts[:, :2])    # x,y untouched — only z displaced
        assert out[:, 2].max() > 0                                 # dent (sign +1) pushes +z
        assert np.abs(out[:, 2]).max() <= amp + 1e-6               # Gaussian peak = 1 at the patch centre

    def test_sign_flips_push_direction(self):
        verts = np.random.RandomState(1).rand(50, 3).astype(np.float32)
        verts[:, 2] = 0.0
        dent, _ = deform(verts, np.random.RandomState(2), sign=1.0)
        bump, _ = deform(verts, np.random.RandomState(2), sign=-1.0)   # same rng -> same patch
        assert dent[:, 2].max() > 0 and bump[:, 2].min() < 0
