from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_template_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    vertices = np.asarray(payload["vertices"], dtype=np.float32)
    faces = np.asarray(payload["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Template vertices must be (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Template faces must be (F, 3), got {faces.shape}")
    return vertices, faces


def mesh_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = [int(x) for x in tri]
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def write_template_mesh_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    if len(mesh.triangles) == 0:
        raise ValueError("Template mesh has no triangles")
    ok = o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"failed to write {path}")

