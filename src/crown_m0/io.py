from __future__ import annotations

from pathlib import Path

import numpy as np


def write_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points[:, :3], fmt="%.6f")


def write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = points[:, :3]
    normals = points[:, 3:6] if points.shape[1] >= 6 else None
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if normals is not None:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        if normals is None:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, n in zip(xyz, normals):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

