from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_BINARY_TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ]
)


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    path: Path

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return vectors / norms


def _parse_ascii_stl(path: Path) -> Mesh:
    vertices: list[list[float]] = []
    face_normals: list[list[float]] = []
    current_normal = [0.0, 0.0, 0.0]
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("facet normal"):
                current_normal = [float(value) for value in stripped.split()[2:5]]
            elif stripped.startswith("vertex"):
                vertices.append([float(value) for value in stripped.split()[1:4]])
                if len(vertices) % 3 == 0:
                    face_normals.append(current_normal)

    vertex_array = np.asarray(vertices, dtype=np.float32)
    faces = np.arange(vertex_array.shape[0], dtype=np.int32).reshape(-1, 3)
    unique_vertices, inverse = np.unique(np.round(vertex_array, 6), axis=0, return_inverse=True)
    unique_faces = inverse.reshape(-1, 3).astype(np.int32)
    normals = np.asarray(face_normals, dtype=np.float32)
    if normals.shape[0] != unique_faces.shape[0]:
        triangles = unique_vertices[unique_faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return Mesh(unique_vertices.astype(np.float32), unique_faces, _normalize_vectors(normals), path)


def _parse_binary_stl(path: Path) -> Mesh:
    payload = path.read_bytes()
    triangle_count = int(np.frombuffer(payload, dtype="<u4", count=1, offset=80)[0])
    triangle_bytes = payload[84:]
    triangles = np.frombuffer(triangle_bytes, dtype=_BINARY_TRIANGLE_DTYPE, count=triangle_count)
    raw_vertices = np.stack([triangles["v0"], triangles["v1"], triangles["v2"]], axis=1).reshape(-1, 3)
    unique_vertices, inverse = np.unique(np.round(raw_vertices, 6), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    normals = triangles["normal"].astype(np.float32)
    zero_mask = np.linalg.norm(normals, axis=1) <= 1e-12
    if np.any(zero_mask):
        tris = unique_vertices[faces[zero_mask]]
        normals[zero_mask] = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return Mesh(unique_vertices.astype(np.float32), faces, _normalize_vectors(normals), path)


def load_mesh(path: str | Path) -> Mesh:
    mesh_path = Path(path)
    payload = mesh_path.read_bytes()
    is_binary = False
    if len(payload) >= 84:
        triangle_count = int(np.frombuffer(payload, dtype="<u4", count=1, offset=80)[0])
        is_binary = 84 + triangle_count * 50 == len(payload)
    if is_binary:
        return _parse_binary_stl(mesh_path)
    return _parse_ascii_stl(mesh_path)


def sample_surface(mesh: Mesh, num_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    triangles = mesh.triangles.astype(np.float32)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    total_area = float(np.sum(areas))
    if total_area <= 1e-12:
        choices = rng.choice(triangles.shape[0], size=num_points, replace=True)
    else:
        probabilities = areas / total_area
        choices = rng.choice(triangles.shape[0], size=num_points, replace=True, p=probabilities)

    chosen = triangles[choices]
    u = rng.random(num_points, dtype=np.float32)
    v = rng.random(num_points, dtype=np.float32)
    mirror = (u + v) > 1.0
    u[mirror] = 1.0 - u[mirror]
    v[mirror] = 1.0 - v[mirror]

    points = chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])
    normals = mesh.face_normals[choices]
    return points.astype(np.float32), normals.astype(np.float32)


def extract_boundary_vertices(mesh: Mesh) -> np.ndarray:
    faces = mesh.faces
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    boundary_vertex_ids = np.unique(boundary_edges.reshape(-1))
    return mesh.vertices[boundary_vertex_ids].astype(np.float32)


def resample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, points.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < num_points
    indices = rng.choice(points.shape[0], size=num_points, replace=replace)
    return points[indices].astype(np.float32)


def write_binary_stl(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    output_path = Path(path)
    triangles = vertices[faces].astype(np.float32)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = _normalize_vectors(normals).astype(np.float32)
    header = b"CrownValidate STL".ljust(80, b" ")
    count = np.uint32(faces.shape[0])
    records = np.zeros(faces.shape[0], dtype=_BINARY_TRIANGLE_DTYPE)
    records["normal"] = normals
    records["v0"] = triangles[:, 0]
    records["v1"] = triangles[:, 1]
    records["v2"] = triangles[:, 2]
    with output_path.open("wb") as handle:
        handle.write(header)
        handle.write(count.tobytes())
        handle.write(records.tobytes())
