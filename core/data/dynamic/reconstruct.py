"""Digital-twin reconstruction (df0.1): a renderable mean-surface mesh per category from the
244 single-view *good* scans.

MVTec is top-down single-view (Zivid): every good instance is captured from the same overhead
pose, so the 244 partials already share a common metric frame — measured cross-frame bbox std
< 0.1 mm in x/y — which means a naive stack (no ICP registration) fuses them into one dense mean
surface. That surface is single-sided (top only) by construction, which is exactly what the twin
needs: the Replicator render reproduces the same top-down view distribution as the real capture,
so no back/underside completion is required. The dominant ground plane is removed so the asset is
object-only, and the largest connected cluster is kept to drop stray table remnants.

CLI:  uv run python -m core.data.dynamic.reconstruct            # all categories
      uv run python -m core.data.dynamic.reconstruct --cat bagel
"""
import argparse

import numpy as np
import open3d as o3d
import tifffile
from jaxtyping import Float

from core.config import Config
from core.data.static.mvtec import Split
from core.obs import Obs


class CategoryReconstructor:
    """Fuse the single-view good partials of one MVTec category into a renderable mean-surface mesh."""

    def __init__(self, voxel: float = 0.0008, poisson_depth: int = 9, plane_dist: float = 0.002,
                 density_trim: float = 0.08, target_faces: int = 150_000):
        self.voxel = voxel
        self.poisson_depth = poisson_depth
        self.plane_dist = plane_dist
        self.density_trim = density_trim
        self.target_faces = target_faces
        self.log = Obs.get()

    @staticmethod
    def categories() -> list[str]:
        root = Config.mvtec_root()
        return sorted(p.name for p in root.iterdir() if (p / Split.TRAIN / "good" / "xyz").is_dir())

    @staticmethod
    def _valid_points(xyz: Float[np.ndarray, "h w 3"]) -> Float[np.ndarray, "n 3"]:
        """Organized position map -> the finite, non-origin points (the sensor writes 0/NaN for misses)."""
        v = np.isfinite(xyz).all(-1) & (np.abs(xyz).sum(-1) > 0)
        return xyz[v]

    def _fuse(self, cat: str) -> o3d.geometry.PointCloud:
        xyz_dir = Config.mvtec_root() / cat / Split.TRAIN / "good" / "xyz"
        chunks = [self._valid_points(tifffile.imread(p).astype(np.float32))
                  for p in sorted(xyz_dir.glob("*.tiff"))]
        pts = np.concatenate(chunks, 0).astype(np.float64)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        return pcd.voxel_down_sample(voxel_size=self.voxel)

    def _isolate_object(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        _, inliers = pcd.segment_plane(self.plane_dist, ransac_n=3, num_iterations=1000)
        obj = pcd.select_by_index(inliers, invert=True)
        labels = np.asarray(obj.cluster_dbscan(eps=self.voxel * 4, min_points=20))
        if labels.max() < 0:
            return obj
        keep = np.flatnonzero(labels == np.bincount(labels[labels >= 0]).argmax())
        return obj.select_by_index(keep.tolist())

    def _poisson(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 4, max_nn=30))
        pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))   # top-down sensor -> -z
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.poisson_depth)
        mesh.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), self.density_trim))
        if len(mesh.triangles) > self.target_faces:
            mesh = mesh.simplify_quadric_decimation(self.target_faces)
        mesh.compute_vertex_normals()
        return mesh

    def build(self, cat: str):
        pcd = self._isolate_object(self._fuse(cat))
        mesh = self._poisson(pcd)
        out_dir = Config.twin_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        obj = out_dir / f"{cat}.obj"
        o3d.io.write_triangle_mesh(str(obj), mesh)
        self.log.info("twin %-12s pts=%d V=%d F=%d -> %s", cat, len(pcd.points),
                      len(mesh.vertices), len(mesh.triangles), obj.name)
        return obj

    def build_all(self) -> list:
        return [self.build(c) for c in self.categories()]

    @staticmethod
    def main():
        ap = argparse.ArgumentParser(description="Reconstruct digital-twin meshes from good scans.")
        ap.add_argument("--cat", default=None, help="single category (default: all)")
        ap.add_argument("--voxel", type=float, default=0.0008,
                        help="fuse voxel size (m); ~Zivid native spacing 0.0002 preserves sub-mm detail")
        ap.add_argument("--poisson-depth", type=int, default=9, help="Poisson octree depth (finer = more detail)")
        ap.add_argument("--target-faces", type=int, default=150_000, help="decimation cap (0 = no decimation)")
        args = ap.parse_args()
        r = CategoryReconstructor(voxel=args.voxel, poisson_depth=args.poisson_depth,
                                  target_faces=args.target_faces or 10**12)
        r.build(args.cat) if args.cat else r.build_all()


if __name__ == "__main__":
    CategoryReconstructor.main()
