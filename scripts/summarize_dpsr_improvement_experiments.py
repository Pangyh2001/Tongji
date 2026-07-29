from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DPSR improvement ablations.")
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--date", default="20260729")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def official_row(label: str, path: Path) -> dict:
    summary = json.loads(
        (path / "test" / "summary_metrics.json").read_text(encoding="utf-8")
    )["dmc_dpsr_marching_cubes"]
    cases = read_csv(path / "test" / "metrics_by_case.csv")
    topology_ok = sum(row.get("topology_ok") == "True" for row in cases)
    return {
        "experiment": label,
        "stl_variant": "zero_level",
        "n": len(cases),
        "point_symmetric_rms": summary.get("point_symmetric_rms_mean"),
        "point_fscore_0p3": summary.get("point_fscore_0p3_mean"),
        "normal_cosine": summary.get("point_normal_cosine_similarity_mean"),
        "stl_symmetric_rms": summary.get("stl_symmetric_rms_mean"),
        "stl_fscore_0p3": summary.get("stl_fscore_0p3_mean"),
        "margin_stl_rms": summary.get("margin_stl_rms_mean"),
        "r1_stl_symmetric_rms": summary.get("r1_stl_symmetric_rms_mean"),
        "topology_ok": topology_ok,
        "topology_failure": len(cases) - topology_ok,
        "topology_ok_rate": topology_ok / max(len(cases), 1),
        "selected_levels": "",
        "path": str(path),
    }


def iso_row(label: str, path: Path, point_source: dict) -> dict:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    cases = read_csv(path / "test" / "metrics_by_case.csv")
    levels = Counter(float(row["selected_level"]) for row in cases if row.get("ok") == "True")
    return {
        "experiment": label,
        "stl_variant": "balanced_iso",
        "n": summary["n"],
        "point_symmetric_rms": point_source["point_symmetric_rms"],
        "point_fscore_0p3": point_source["point_fscore_0p3"],
        "normal_cosine": point_source["normal_cosine"],
        "stl_symmetric_rms": summary["stl_symmetric_rms_mean"],
        "stl_fscore_0p3": summary["stl_fscore_0p3_mean"],
        "margin_stl_rms": summary["margin_stl_rms_mean"],
        "r1_stl_symmetric_rms": summary["r1_stl_symmetric_rms_mean"],
        "topology_ok": summary["topology_ok"],
        "topology_failure": summary["topology_failures"],
        "topology_ok_rate": summary["topology_ok"] / max(summary["n"], 1),
        "selected_levels": json.dumps(dict(sorted(levels.items())), ensure_ascii=False),
        "path": str(path),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    current = args.result_root / args.date
    baseline_path = args.result_root / "20260728" / "m2_mla_dmc_dpsr128_grid100"
    rows = []

    baseline = official_row("M2 baseline", baseline_path)
    rows.append(baseline)
    rows.append(
        iso_row(
            "M2 + E0 balanced iso",
            current / "e0c_m2_iso_balanced",
            baseline,
        )
    )

    experiments = [
        ("E1 margin zero", "e1_m2_margin_zero", "e1b_m2_margin_zero_iso_balanced"),
        ("E2 detail", "e2_m2_detail", "e2b_m2_detail_iso_balanced"),
        ("E3 topology", "e3_m2_topology", "e3b_m2_topology_iso_balanced"),
        ("E4 combined", "e4_m2_combined", "e4b_m2_combined_iso_balanced"),
    ]
    for label, official_name, iso_name in experiments:
        zero = official_row(label, current / official_name)
        rows.append(zero)
        rows.append(iso_row(f"{label} + balanced iso", current / iso_name, zero))

    write_csv(current / "dpsr_improvement_summary.csv", rows)
    (current / "dpsr_improvement_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# DPSR improvement experiments",
        "",
        "| Experiment | STL variant | STL RMS | STL F-score | Margin STL RMS | R1 STL RMS | Topology |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['stl_variant']} | "
            f"{row['stl_symmetric_rms']:.3f} | {row['stl_fscore_0p3']:.3f} | "
            f"{row['margin_stl_rms']:.3f} | {row['r1_stl_symmetric_rms']:.3f} | "
            f"{row['topology_ok']}/{row['n']} |"
        )
    lines.extend(
        [
            "",
            "The balanced iso-level selector does not use GT. It first requires a "
            "watertight genus-0 mesh, then minimizes predicted-point adherence plus "
            "0.25 times margin adherence.",
            "",
        ]
    )
    (current / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
