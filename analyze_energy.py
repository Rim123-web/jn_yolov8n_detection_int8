#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics

from common import ROOT, append_timeline, percentile, write_csv, write_json


def numeric(rows, field):
    return [float(row[field]) for row in rows if row.get(field) not in (None, "", "NA")]


def main():
    append_timeline("energy_resource_analysis", "START", "timestamped trapezoidal integration")
    active = list(csv.DictReader((ROOT / "monitoring/active_samples.csv").open(encoding="utf-8")))
    idle_summary = json.loads((ROOT / "monitoring/idle_summary.json").read_text())
    provenance = json.loads((ROOT / "raw/execution_count_provenance.json").read_text())
    start = float(provenance["official_wall_start_monotonic_s"])
    end = float(provenance["official_wall_end_monotonic_s"])
    duration = end - start
    active = sorted(active, key=lambda row: float(row["monotonic_s"]))
    if len(active) < 2 or not all(str(row["valid"]).lower() == "true" for row in active):
        raise RuntimeError("Active telemetry is incomplete or invalid")
    idle_power = float(idle_summary["power_mean_w"])
    integration_rows = []
    points = [(start, float(active[0]["power_tot_w"]))] + [(float(row["monotonic_s"]), float(row["power_tot_w"])) for row in active] + [(end, float(active[-1]["power_tot_w"]))]
    gross_energy = 0.0
    net_energy = 0.0
    for index, ((t0, p0), (t1, p1)) in enumerate(zip(points, points[1:]), 1):
        delta = max(0.0, t1 - t0)
        gross = 0.5 * (p0 + p1) * delta
        n0, n1 = max(0.0, p0 - idle_power), max(0.0, p1 - idle_power)
        net = 0.5 * (n0 + n1) * delta
        gross_energy += gross
        net_energy += net
        integration_rows.append({"interval_index": index, "start_monotonic_s": t0, "end_monotonic_s": t1, "duration_s": delta, "start_power_w": p0, "end_power_w": p1, "gross_energy_j": gross, "idle_power_w": idle_power, "idle_corrected_energy_j": net})
    write_csv(ROOT / "energy/power_integration.csv", integration_rows)
    powers = numeric(active, "power_tot_w")
    cpus = numeric(active, "cpu_utilization_percent")
    gpus = numeric(active, "gpu_utilization_percent")
    ram = numeric(active, "ram_used_mib")
    rss = numeric(active, "process_rss_mib")
    temps = numeric(active, "temperature_c")
    throttles = [row for row in active if str(row.get("throttling_status", "NA")).upper() not in {"NA", "NONE", "0", "FALSE", "NOT THROTTLED"}]
    before_oom = (ROOT / "logs/kernel_oom_before.txt").read_text(encoding="utf-8", errors="replace").splitlines() if (ROOT / "logs/kernel_oom_before.txt").exists() else []
    after_oom = (ROOT / "logs/kernel_oom_after.txt").read_text(encoding="utf-8", errors="replace").splitlines() if (ROOT / "logs/kernel_oom_after.txt").exists() else []
    before_set = set(before_oom)
    new_oom = [line for line in after_oom if line not in before_set]
    coverage = (float(active[-1]["monotonic_s"]) - float(active[0]["monotonic_s"])) / duration if duration else 0.0
    summary = {
        "status": "PASS", "method": "trapezoidal integration over timestamped Power TOT; benchmark endpoints extended using nearest valid sample",
        "measured_pass_duration_s": duration, "telemetry_sample_count": len(active), "telemetry_internal_coverage_fraction": coverage,
        "active_mean_power_w_time_weighted": gross_energy / duration, "active_sample_mean_power_w": statistics.fmean(powers),
        "active_median_power_w": statistics.median(powers), "active_p95_power_w": percentile(powers, 95),
        "active_max_power_w": max(powers), "active_min_power_w": min(powers),
        "gross_energy_j": gross_energy, "gross_energy_per_image_j": gross_energy / 192,
        "idle_mean_power_w": idle_power, "idle_corrected_active_power_w": net_energy / duration,
        "idle_corrected_energy_j": net_energy, "idle_corrected_energy_per_image_j": net_energy / 192,
        "power_outlier_warning": max(powers) > 30.0, "warmups_excluded": True,
    }
    if summary["power_outlier_warning"]:
        summary["status"] = "FAIL"
    write_json(ROOT / "energy/energy_summary.json", summary)
    write_csv(ROOT / "energy/energy_per_image_summary.csv", [{"images": 192, "gross_energy_j": gross_energy, "gross_energy_per_image_j": gross_energy / 192, "idle_corrected_energy_j": net_energy, "idle_corrected_energy_per_image_j": net_energy / 192}])
    resource = {
        "status": summary["status"], "cpu_mean_percent": statistics.fmean(cpus), "cpu_max_percent": max(cpus),
        "gpu_mean_percent": statistics.fmean(gpus), "gpu_max_percent": max(gpus),
        "ram_mean_mib": statistics.fmean(ram), "ram_max_mib": max(ram),
        "process_rss_mean_mib": statistics.fmean(rss), "process_rss_max_mib": max(rss),
        "sample_count": len(active), "telemetry_method": "jtop/jetson-stats Power TOT",
    }
    write_json(ROOT / "resources/resource_summary.json", resource)
    write_json(ROOT / "temperature/temperature_summary.json", {"status": "PASS", "mean_c": statistics.fmean(temps), "max_c": max(temps), "min_c": min(temps), "p95_c": percentile(temps, 95)})
    write_json(ROOT / "throttling/throttling_summary.json", {
        "status": "PASS" if not throttles and not new_oom else "FAIL",
        "event_count": len(throttles), "events": throttles,
        "oom_events": len(new_oom), "new_oom_evidence": new_oom,
        "oom_evidence_before": "logs/kernel_oom_before.txt",
        "oom_evidence_after": "logs/kernel_oom_after.txt",
    })
    append_timeline("energy_resource_analysis", summary["status"], f"gross={gross_energy:.3f}J net={net_energy:.3f}J")
    print(json.dumps({"energy": summary, "resources": resource}, indent=2))
    if summary["status"] != "PASS" or throttles or new_oom:
        raise RuntimeError("Energy/resource validation failed")


if __name__ == "__main__":
    main()
