from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create M0 prediction visualizations.")
    parser.add_argument("--prediction-dir", type=Path, default=Path("predictions/m0_new_all"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/m0_new_test"))
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all selected cases.")
    parser.add_argument("--sample-points", type=int, default=6000)
    parser.add_argument("--error-clip-mm", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cases = select_cases(args)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    png_dir = args.output_dir / "png"
    ply_dir = args.output_dir / "error_ply"
    png_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_dir in tqdm(cases, desc="visualize"):
        case_path = Path(case_dir)
        pred_path = args.prediction_dir / f"{safe_case_name(case_path, args.data_dir)}.npy"
        gt_path = case_path / "train" / "crown_points.npy"
        if not pred_path.exists() or not gt_path.exists():
            rows.append({"case": case_dir, "ok": False, "error": "missing prediction or gt"})
            continue

        pred = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        pred_xyz = pred[:, :3]
        gt_xyz = gt[:, :3]

        dist_pred_to_gt, _ = cKDTree(gt_xyz).query(pred_xyz, k=1)
        dist_gt_to_pred, _ = cKDTree(pred_xyz).query(gt_xyz, k=1)
        metrics = {
            "case": case_dir,
            "ok": True,
            "pred_to_gt_mean": float(np.mean(dist_pred_to_gt)),
            "pred_to_gt_rms": float(np.sqrt(np.mean(dist_pred_to_gt**2))),
            "pred_to_gt_hd95": float(np.percentile(dist_pred_to_gt, 95)),
            "gt_to_pred_mean": float(np.mean(dist_gt_to_pred)),
            "gt_to_pred_rms": float(np.sqrt(np.mean(dist_gt_to_pred**2))),
            "gt_to_pred_hd95": float(np.percentile(dist_gt_to_pred, 95)),
        }
        rows.append(metrics)

        stem = safe_case_name(case_path, args.data_dir)
        write_error_ply(ply_dir / f"{stem}_pred_error.ply", pred_xyz, dist_pred_to_gt, args.error_clip_mm)
        make_png(
            png_dir / f"{stem}.png",
            gt_xyz,
            pred_xyz,
            dist_pred_to_gt,
            title=stem,
            metrics=metrics,
            sample_points=args.sample_points,
            error_clip_mm=args.error_clip_mm,
            rng=rng,
        )

    (args.output_dir / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics_csv(args.output_dir / "metrics.csv", rows)
    write_index(args.output_dir / "index.html", rows, png_dir)
    print(f"wrote {args.output_dir}")


def select_cases(args: argparse.Namespace) -> list[str]:
    if args.split == "all":
        return [str(p.parent) for p in sorted(args.data_dir.rglob("train/metadata.json"))]
    payload = json.loads(args.split_file.read_text(encoding="utf-8"))
    return payload[args.split]


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        rel = case_path
    return "__".join(rel.parts)


def make_png(
    path: Path,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    pred_error: np.ndarray,
    *,
    title: str,
    metrics: dict,
    sample_points: int,
    error_clip_mm: float,
    rng: np.random.Generator,
) -> None:
    gt_sample = sample_rows(gt_xyz, sample_points, rng)
    pred_idx = sample_indices(pred_xyz.shape[0], sample_points, rng)
    pred_sample = pred_xyz[pred_idx]
    err_sample = pred_error[pred_idx]

    mins = np.minimum(gt_xyz.min(axis=0), pred_xyz.min(axis=0))
    maxs = np.maximum(gt_xyz.max(axis=0), pred_xyz.max(axis=0))
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    bounds = np.stack([center - span / 2.0, center + span / 2.0])

    fig = plt.figure(figsize=(15, 5), dpi=160)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    axes[0].scatter(gt_sample[:, 0], gt_sample[:, 1], gt_sample[:, 2], s=1, c="#4c78a8", alpha=0.75)
    axes[0].set_title("Technician crown")

    axes[1].scatter(pred_sample[:, 0], pred_sample[:, 1], pred_sample[:, 2], s=1, c="#f58518", alpha=0.75)
    axes[1].set_title("M0 prediction")

    sc = axes[2].scatter(
        pred_sample[:, 0],
        pred_sample[:, 1],
        pred_sample[:, 2],
        s=1,
        c=np.clip(err_sample, 0, error_clip_mm),
        cmap="turbo",
        vmin=0,
        vmax=error_clip_mm,
        alpha=0.9,
    )
    axes[2].set_title("Pred error to GT")
    fig.colorbar(sc, ax=axes[2], shrink=0.62, pad=0.02, label="mm")

    for ax in axes:
        ax.set_xlim(bounds[0, 0], bounds[1, 0])
        ax.set_ylim(bounds[0, 1], bounds[1, 1])
        ax.set_zlim(bounds[0, 2], bounds[1, 2])
        ax.view_init(elev=25, azim=-65)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.suptitle(
        f"{title} | RMS {metrics['pred_to_gt_rms']:.3f} mm | HD95 {metrics['pred_to_gt_hd95']:.3f} mm",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def sample_indices(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if n <= count:
        return np.arange(n)
    return rng.choice(n, size=count, replace=False)


def sample_rows(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    return points[sample_indices(points.shape[0], count, rng)]


def write_error_ply(path: Path, xyz: np.ndarray, error: np.ndarray, clip_mm: float) -> None:
    colors = error_to_rgb(error, clip_mm)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def error_to_rgb(error: np.ndarray, clip_mm: float) -> np.ndarray:
    values = np.clip(error / max(clip_mm, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    rgba = cmap(values)
    return (rgba[:, :3] * 255).astype(np.uint8)


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    import csv

    keys = [
        "case",
        "ok",
        "pred_to_gt_mean",
        "pred_to_gt_rms",
        "pred_to_gt_hd95",
        "gt_to_pred_mean",
        "gt_to_pred_rms",
        "gt_to_pred_hd95",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_index(path: Path, rows: list[dict], png_dir: Path) -> None:
    ok_rows = [r for r in rows if r.get("ok")]
    ok_rows.sort(key=lambda r: r.get("pred_to_gt_rms", 0), reverse=True)
    rel_png = png_dir.relative_to(path.parent)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>M0 visualizations</title>",
        "<style>body{font-family:sans-serif;margin:24px} img{width:100%;max-width:1200px;border:1px solid #ddd} .case{margin:28px 0}</style>",
        "<h1>M0 visualizations</h1>",
        "<p>Sorted by pred-to-GT RMS descending.</p>",
    ]
    for row in ok_rows:
        stem = safe_case_name(Path(row["case"]), Path("data"))
        parts.append("<div class='case'>")
        parts.append(
            f"<h2>{stem}</h2><p>RMS {row['pred_to_gt_rms']:.3f} mm, HD95 {row['pred_to_gt_hd95']:.3f} mm</p>"
        )
        parts.append(f"<img src='{rel_png}/{stem}.png'>")
        parts.append("</div>")
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
