"""Lossless metric values and reproducible configuration in UTF-8 CSV reports."""

import csv
import json
import math
from pathlib import Path
from statistics import mean

REPORT_FIELDS = (
    "filename", "algorithm", "status", "parameters", "input_path", "output_path",
    "before_PSNR", "before_SSIM", "PSNR", "SSIM", "delta_PSNR", "delta_SSIM",
    "time", "error", "metrics_error",
)


def metric_delta(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    return {key: 0.0 if before[key] == after[key] else after[key] - before[key]
            for key in ("PSNR", "SSIM")}


def report_row(filename: str, algorithm: str, data: dict, input_path="", output_path="") -> dict:
    row = {"filename": filename, "algorithm": algorithm,
           "status": data.get("status", "success"),
           "parameters": json.dumps(data.get("parameters", {}), ensure_ascii=False, sort_keys=True),
           "input_path": str(input_path), "output_path": str(output_path),
           "time": data.get("time", ""), "error": data.get("error", ""),
           "metrics_error": data.get("metrics_error", "")}
    for source, prefix in (("baseline", "before_"), ("metrics", ""), ("delta", "delta_")):
        for key in ("PSNR", "SSIM"):
            row[prefix + key] = (data.get(source) or {}).get(key, "")
    return row


def write_csv(path: str | Path, rows: list[dict], fields=REPORT_FIELDS) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for algorithm in dict.fromkeys(row["algorithm"] for row in rows):
        group = [r for r in rows if r["algorithm"] == algorithm]
        evaluated = [r for r in group if isinstance(r["PSNR"], (float, int))]
        summary = {"algorithm": algorithm, "attempted": len(group),
                   "processed": sum(r["status"] == "success" for r in group),
                   "skipped": sum(r["status"] == "skipped" for r in group),
                   "failed": sum(r["status"] == "failed" for r in group),
                   "evaluated": len(evaluated)}
        for key in ("before_PSNR", "before_SSIM", "PSNR", "SSIM", "delta_PSNR", "delta_SSIM"):
            values = [r[key] for r in evaluated]
            # Opposite infinities have no arithmetic mean; keep CSV explicit.
            summary["mean_" + key] = ("undefined" if any(v == math.inf for v in values)
                                      and any(v == -math.inf for v in values)
                                      else mean(values) if values else "")
        for key in ("PSNR", "SSIM"):
            summary["improved_" + key] = sum(r["delta_" + key] > 0 for r in evaluated)
        summaries.append(summary)
    return summaries
