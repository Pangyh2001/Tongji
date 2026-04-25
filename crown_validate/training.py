from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import CrownCaseDataset, collate_cases
from .losses import compute_loss
from .metrics import compute_case_metrics
from .model import CrownDeformationNet
from .stl import write_binary_stl
from .utils import ensure_dir, json_dump


def _move_batch_to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    moved: dict[str, torch.Tensor | list[str]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _evaluate_loader(
    model: CrownDeformationNet,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, float | bool | int],
    metric_cfg: dict[str, float],
    output_dir: Path | None = None,
) -> tuple[float, list[dict[str, float | str]]]:
    model.eval()
    all_losses: list[float] = []
    results: list[dict[str, float | str]] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch["prep_points"], batch["opposing_points"])
            loss, breakdown = compute_loss(outputs, batch, loss_cfg)
            all_losses.append(float(loss.detach().cpu()))

            pred_points = outputs["pred_points"].detach().cpu().numpy()
            pred_margin = outputs["pred_margin"].detach().cpu().numpy()
            gt_points = batch["crown_points"].detach().cpu().numpy()
            margin_points = batch["margin_points"].detach().cpu().numpy()
            centers = batch["center"].detach().cpu().numpy()
            scales = batch["scale"].detach().cpu().numpy()
            case_ids = batch["case_id"]

            for item_index, case_id in enumerate(case_ids):
                center = centers[item_index]
                scale = float(scales[item_index][0])
                pred_points_mm = pred_points[item_index] * scale + center
                pred_margin_mm = pred_margin[item_index] * scale + center
                gt_points_mm = gt_points[item_index] * scale + center
                gt_margin_mm = margin_points[item_index] * scale + center if margin_points[item_index].size else None
                metrics = compute_case_metrics(
                    pred_points_mm=pred_points_mm,
                    gt_points_mm=gt_points_mm,
                    pred_margin_mm=pred_margin_mm,
                    gt_margin_mm=gt_margin_mm,
                    fscore_threshold_mm=float(metric_cfg["fscore_threshold_mm"]),
                )
                metrics_row: dict[str, float | str] = {"case_id": case_id, **metrics, **breakdown}
                results.append(metrics_row)

                if output_dir is not None:
                    case_dir = ensure_dir(output_dir / case_id)
                    np.save(case_dir / "pred_points_mm.npy", pred_points_mm.astype(np.float32))
                    np.save(case_dir / "pred_margin_mm.npy", pred_margin_mm.astype(np.float32))
                    write_binary_stl(case_dir / "pred_crown.stl", pred_points_mm.astype(np.float32), model.template_faces.detach().cpu().numpy())
    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    return mean_loss, results


def train_single_fold(
    model: CrownDeformationNet,
    processed_dir: str | Path,
    train_case_ids: list[str],
    val_case_ids: list[str],
    test_case_ids: list[str],
    device: torch.device,
    loss_cfg: dict[str, float | bool | int],
    training_cfg: dict[str, float | int],
    metric_cfg: dict[str, float],
    output_dir: str | Path,
    run_name: str | None = None,
) -> dict[str, float | str]:
    fold_dir = ensure_dir(output_dir)
    train_dataset = CrownCaseDataset(processed_dir, train_case_ids)
    val_dataset = CrownCaseDataset(processed_dir, val_case_ids)
    test_dataset = CrownCaseDataset(processed_dir, test_case_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        collate_fn=collate_cases,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        collate_fn=collate_cases,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        collate_fn=collate_cases,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    best_state = None
    best_val_loss = float("inf")
    patience_left = int(training_cfg["patience"])
    history_rows: list[dict[str, float | int]] = []
    display_name = run_name or fold_dir.name

    print(
        f"[{display_name}] start "
        f"train_cases={train_case_ids} val_cases={val_case_ids} test_cases={test_case_ids}",
        flush=True,
    )

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["prep_points"], batch["opposing_points"])
            loss, _ = compute_loss(outputs, batch, loss_cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(training_cfg["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss, _ = _evaluate_loader(model, val_loader, device, loss_cfg, metric_cfg)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = int(training_cfg["patience"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
        print(
            f"[{display_name}] "
            f"epoch {epoch}/{int(training_cfg['epochs'])} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"best_val={best_val_loss:.6f} "
            f"patience_left={patience_left} "
            f"{'improved' if improved else ''}".rstrip(),
            flush=True,
        )
        if patience_left <= 0:
            print(f"[{display_name}] early stopping at epoch {epoch}", flush=True)
            break

    history_path = fold_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history_rows)

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")
    torch.save(best_state, fold_dir / "checkpoint_best.pt")
    model.load_state_dict(best_state)

    _, val_results = _evaluate_loader(model, val_loader, device, loss_cfg, metric_cfg, output_dir=fold_dir / "val_predictions")
    _, test_results = _evaluate_loader(model, test_loader, device, loss_cfg, metric_cfg, output_dir=fold_dir / "test_predictions")

    pd.DataFrame(val_results).to_csv(fold_dir / "val_metrics.csv", index=False)
    pd.DataFrame(test_results).to_csv(fold_dir / "test_metrics.csv", index=False)

    if not test_results:
        raise RuntimeError("No test results were produced.")
    fold_summary = dict(test_results[0])
    fold_summary["best_val_loss"] = best_val_loss
    json_dump({key: (float(value) if isinstance(value, (int, float, np.floating)) else value) for key, value in fold_summary.items()}, fold_dir / "summary.json")
    print(
        f"[{display_name}] done "
        f"best_val={best_val_loss:.6f} "
        f"test_chamfer={float(fold_summary['chamfer_l2_mm2']):.6f} "
        f"test_hd95={float(fold_summary['hd95_mm']):.6f}",
        flush=True,
    )
    return fold_summary
