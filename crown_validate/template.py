from __future__ import annotations

import math

import numpy as np


def build_open_cap_template(
    num_latitude_rings: int,
    num_longitude_segments: int,
    theta_max_radians: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = [[0.0, 0.0, 1.0]]
    for ring_idx in range(1, num_latitude_rings + 1):
        theta = theta_max_radians * ring_idx / num_latitude_rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for lon_idx in range(num_longitude_segments):
            phi = 2.0 * math.pi * lon_idx / num_longitude_segments
            vertices.append(
                [
                    sin_theta * math.cos(phi),
                    sin_theta * math.sin(phi),
                    cos_theta,
                ]
            )

    faces: list[list[int]] = []
    first_ring_start = 1
    for lon_idx in range(num_longitude_segments):
        next_lon = (lon_idx + 1) % num_longitude_segments
        faces.append([0, first_ring_start + next_lon, first_ring_start + lon_idx])

    for ring_idx in range(1, num_latitude_rings):
        current_start = 1 + (ring_idx - 1) * num_longitude_segments
        next_start = current_start + num_longitude_segments
        for lon_idx in range(num_longitude_segments):
            next_lon = (lon_idx + 1) % num_longitude_segments
            a = current_start + lon_idx
            b = current_start + next_lon
            c = next_start + lon_idx
            d = next_start + next_lon
            faces.append([a, b, c])
            faces.append([b, d, c])

    boundary_start = 1 + (num_latitude_rings - 1) * num_longitude_segments
    boundary_indices = np.arange(boundary_start, boundary_start + num_longitude_segments, dtype=np.int64)
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
        boundary_indices,
    )
