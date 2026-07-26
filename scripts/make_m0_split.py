from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a patient-level split for crown M0 experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--attempts", type=int, default=2000, help="Random patient-level split candidates to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_records(args.data_dir)
    if not records:
        raise SystemExit(f"No train/metadata.json records found under {args.data_dir}")

    split = patient_level_split(
        records,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        attempts=args.attempts,
    )
    summary = summarize_split(split)
    payload = {
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "fractions": {"train": args.train_fraction, "val": args.val_fraction, "test": args.test_fraction},
        "summary": summary,
        "train": [r["case_dir"] for r in split["train"]],
        "val": [r["case_dir"] for r in split["val"]],
        "test": [r["case_dir"] for r in split["test"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def discover_records(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    for metadata_path in sorted(data_dir.rglob("train/metadata.json")):
        train_dir = metadata_path.parent
        case_dir = train_dir.parent
        if not all((train_dir / name).exists() for name in ("prep_points.npy", "antagonist_points.npy", "crown_points.npy")):
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        tooth_id = str(meta.get("tooth_id") or infer_tooth_id(case_dir))
        patient_id = infer_patient_id(case_dir, meta)
        arch = str(meta.get("role_assignment", {}).get("prep_arch", "unknown")).lower()
        records.append(
            {
                "case_dir": str(case_dir),
                "patient_id": patient_id,
                "tooth_id": tooth_id,
                "arch": arch,
                "stratum": f"{tooth_group(tooth_id)}_{arch}",
            }
        )
    return records


def patient_level_split(records: list[dict], *, train_fraction: float, val_fraction: float, test_fraction: float, seed: int, attempts: int) -> dict[str, list[dict]]:
    import random

    fractions = {"train": train_fraction, "val": val_fraction, "test": test_fraction}
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {fractions}")

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_patient[record["patient_id"]].append(record)

    total_cases = len(records)
    target_test = round(total_cases * test_fraction)
    target_val = round(total_cases * val_fraction)
    global_tooth = Counter(r["tooth_id"] for r in records)
    global_arch = Counter(r["arch"] for r in records)

    best_split: dict[str, list[dict]] | None = None
    best_score = float("inf")
    patient_items = list(by_patient.items())

    for attempt in range(max(1, attempts)):
        rng = random.Random(seed + attempt)
        patients = patient_items[:]
        rng.shuffle(patients)
        split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        test_cases = 0
        val_cases = 0

        for _, patient_records in patients:
            group_size = len(patient_records)
            if test_cases < target_test and abs((test_cases + group_size) - target_test) <= abs(test_cases - target_test):
                split["test"].extend(patient_records)
                test_cases += group_size
            elif val_cases < target_val and abs((val_cases + group_size) - target_val) <= abs(val_cases - target_val):
                split["val"].extend(patient_records)
                val_cases += group_size
            else:
                split["train"].extend(patient_records)

        score = split_score(split, global_tooth, global_arch, fractions)
        if score < best_score:
            best_score = score
            best_split = split

    assert best_split is not None
    for name in best_split:
        best_split[name].sort(key=lambda r: r["case_dir"])
    return best_split


def split_score(split: dict[str, list[dict]], global_tooth: Counter, global_arch: Counter, fractions: dict[str, float]) -> float:
    score = 0.0
    total = sum(len(v) for v in split.values())
    for name, records in split.items():
        frac = fractions[name]
        count = len(records)
        score += 5.0 * abs(count - total * frac) / max(total * frac, 1.0)

        tooth_counts = Counter(r["tooth_id"] for r in records)
        for tooth, total_tooth in global_tooth.items():
            expected = total_tooth * frac
            observed = tooth_counts.get(tooth, 0)
            # Rare teeth should appear when possible, but not dominate the objective.
            score += 0.35 * abs(observed - expected) / max(expected, 1.0)

        arch_counts = Counter(r["arch"] for r in records)
        for arch, total_arch in global_arch.items():
            expected = total_arch * frac
            observed = arch_counts.get(arch, 0)
            score += 0.2 * abs(observed - expected) / max(expected, 1.0)
    score += patient_leakage_penalty(split)
    return score


def patient_leakage_penalty(split: dict[str, list[dict]]) -> float:
    seen: dict[str, str] = {}
    penalty = 0.0
    for name, records in split.items():
        for record in records:
            patient_id = record["patient_id"]
            if patient_id in seen and seen[patient_id] != name:
                penalty += 1000.0
            seen[patient_id] = name
    return penalty


def summarize_split(split: dict[str, list[dict]]) -> dict:
    summary = {}
    for name, records in split.items():
        summary[name] = {
            "cases": len(records),
            "patients": len({r["patient_id"] for r in records}),
            "tooth_distribution": dict(sorted(Counter(r["tooth_id"] for r in records).items())),
            "arch_distribution": dict(sorted(Counter(r["arch"] for r in records).items())),
            "stratum_distribution": dict(sorted(Counter(r["stratum"] for r in records).items())),
        }
    return summary


def infer_tooth_id(case_dir: Path) -> str:
    if case_dir.name.isdigit():
        return case_dir.name
    match = re.search(r"_(\d{2})$", case_dir.name)
    if not match:
        raise ValueError(f"Cannot infer tooth id from {case_dir}")
    return match.group(1)


def infer_patient_id(case_dir: Path, meta: dict) -> str:
    if case_dir.name.isdigit():
        return case_dir.parent.name
    case_name = str(meta.get("case") or case_dir.name)
    match = re.match(r"(.+)_\d{2}$", case_name)
    return match.group(1) if match else case_name


def tooth_group(tooth_id: str) -> str:
    number = int(tooth_id)
    last = number % 10
    arch = "upper" if number < 30 else "lower"
    kind = "premolar" if last in (4, 5) else "molar"
    return f"{arch}_{kind}"


if __name__ == "__main__":
    main()
