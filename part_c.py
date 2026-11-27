from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
import stim
from scipy.sparse import csc_matrix
from scipy.stats import binomtest
from ldpc import BpOsdDecoder

import part_b as b


# ---------- Experiment definitions ----------

REGIMES = {
    "uniform": {
        "p1": 1.0, "p2": 1.0, "pm": 1.0, "pr": 1.0, "pi": 1.0,
    },
    "idle_dominated": {
        "p1": 0.1, "p2": 0.1, "pm": 0.1, "pr": 0.1, "pi": 1.0,
    },
    "gate_dominated": {
        "p1": 0.1, "p2": 1.0, "pm": 0.1, "pr": 0.1, "pi": 0.01,
    },
    "measurement_dominated": {
        "p1": 0.1, "p2": 0.1, "pm": 1.0, "pr": 0.1, "pi": 0.1,
    },
}

CODE_FACTORIES = {
    "surface_d3": lambda: b.surface_code(3),
    "surface_d5": lambda: b.surface_code(5),
    "bb72": b.bb72_code,
}


def noise_for(regime, p):
    factors = REGIMES[regime]
    return b.Noise(**{
        name: factor * p
        for name, factor in factors.items()
    })


def digest_object(obj):
    text = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


# ---------- DEM conversion ----------

def toggle(indices, value):
    """Toggle one binary index: duplicate effects cancel modulo two."""
    if value in indices:
        indices.remove(value)
    else:
        indices.add(value)


def dem_to_matrices(dem):
    """
    Build sparse fault-to-detector and fault-to-observable matrices.

    Merge only identical (detector effects, observable effects).
    Do not split correlated components at '^'.
    """
    merged = {}

    for instruction in dem.flattened():
        if instruction.type in ("detector", "logical_observable"):
            continue

        if instruction.type != "error":
            raise ValueError(
                f"Unexpected flattened DEM instruction: {instruction.type}"
            )

        # args_copy()[0] is the FIRST argument: the probability.
        probability = float(instruction.args_copy()[0])

        if probability == 0:
            continue

        if not 0 < probability <= 0.5:
            raise ValueError(
                "This benchmark expects DEM mechanism probabilities "
                f"in (0, 0.5]; got {probability}"
            )

        detectors = set()
        observables = set()

        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                toggle(detectors, int(target.val))
            elif target.is_logical_observable_id():
                toggle(observables, int(target.val))
            elif target.is_separator():
                # Components remain part of the SAME random mechanism.
                continue
            else:
                raise ValueError(f"Unexpected DEM target: {target}")

        key = (
            tuple(sorted(detectors)),
            tuple(sorted(observables)),
        )

        old_probability = merged.get(key, 0.0)

        # XOR probability for independent identical mechanisms.
        merged[key] = (
            old_probability + probability
            - 2 * old_probability * probability
        )

    columns = []
    invisible = []

    for (detectors, observables), probability in sorted(merged.items()):
        if not detectors:
            if observables:
                invisible.append({
                    "observables": list(observables),
                    "probability": probability,
                })
            # No detector information identifies this independent event.
            # Since probability <= 0.5, predict that it did not occur.
            # Its actual effect remains present in CIRCUIT samples.
            continue

        columns.append((detectors, observables, probability))

    h_rows, h_cols = [], []
    o_rows, o_cols = [], []
    probabilities = []

    for col, (detectors, observables, probability) in enumerate(columns):
        for row in detectors:
            h_rows.append(row)
            h_cols.append(col)

        for row in observables:
            o_rows.append(row)
            o_cols.append(col)

        probabilities.append(probability)

    ncols = len(columns)

    h = csc_matrix(
        (
            np.ones(len(h_rows), dtype=np.uint8),
            (h_rows, h_cols),
        ),
        shape=(dem.num_detectors, ncols),
        dtype=np.uint8,
    )

    o = csc_matrix(
        (
            np.ones(len(o_rows), dtype=np.uint8),
            (o_rows, o_cols),
        ),
        shape=(dem.num_observables, ncols),
        dtype=np.uint8,
    )

    return h, o, np.array(probabilities, dtype=float), invisible


# ---------- Decoder ----------

class CircuitBPOSD:
    def __init__(self, dem, settings):
        self.h, self.o, self.priors, self.invisible = dem_to_matrices(dem)
        self.engine = None

        if self.h.shape[1] > 0:
            self.engine = BpOsdDecoder(
                self.h,
                error_channel=self.priors.tolist(),
                max_iter=settings["max_bp_iters"],
                bp_method=settings["bp_method"],
                ms_scaling_factor=0.625,
                schedule="parallel",
                omp_thread_count=1,
                input_vector_type="syndrome",
                osd_method=(
                    "OSD_0"
                    if settings["osd_order"] == 0
                    else "OSD_CS"
                ),
                osd_order=settings["osd_order"],
            )

    def decode_batch(self, detectors):
        detectors = np.asarray(detectors, dtype=np.uint8)

        if detectors.ndim != 2:
            raise ValueError("Expected a 2D array of detector samples")

        if detectors.shape[1] != self.h.shape[0]:
            raise ValueError("Wrong number of detector columns")

        predictions = np.zeros(
            (detectors.shape[0], self.o.shape[0]),
            dtype=np.bool_,
        )

        for shot_index, syndrome in enumerate(detectors):
            # No detectors: choose the no-fault estimate.
            if not syndrome.any():
                continue

            if self.engine is None:
                raise RuntimeError(
                    "Nonzero syndrome with no detectable fault columns"
                )

            correction = np.asarray(
                self.engine.decode(syndrome.copy()),
                dtype=np.int64,
            ).reshape(-1)

            if correction.shape != (self.h.shape[1],):
                raise RuntimeError("Unexpected decoder output shape")

            predicted_syndrome = (
                np.asarray(self.h @ correction).reshape(-1) % 2
            )

            if not np.array_equal(predicted_syndrome, syndrome):
                raise RuntimeError(
                    "BP-OSD returned a syndrome-inconsistent correction. "
                    "The run stops instead of discarding this shot."
                )

            predictions[shot_index] = (
                np.asarray(self.o @ correction).reshape(-1) % 2
            ).astype(np.bool_)

        return predictions


# ---------- Small decoder/converter tests ----------

def self_test():
    model = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.2) D0 L0
        error(0.05) D0 L1
        error(0.03) L1
        error(0) D1
        """
    )

    h, o, probabilities, invisible = dem_to_matrices(model)

    # Two distinct logical effects must remain separate columns,
    # even though their detector effects agree.
    assert h.shape == (2, 2)
    assert o.shape == (2, 2)

    assert np.array_equal(
        h.toarray(),
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
    )
    assert np.array_equal(
        o.toarray(),
        np.eye(2, dtype=np.uint8),
    )
    assert np.allclose(probabilities, [0.26, 0.05])
    assert len(invisible) == 1

    # A separator must not create two independent columns.
    correlated = stim.DetectorErrorModel(
        "error(0.04) D0 ^ D1 L0"
    )
    h2, o2, p2, _ = dem_to_matrices(correlated)

    assert h2.shape == (2, 1)
    assert np.array_equal(
        h2.toarray(),
        np.array([[1], [1]], dtype=np.uint8),
    )
    assert np.array_equal(
        o2.toarray(),
        np.array([[1]], dtype=np.uint8),
    )

    # End-to-end decoder API test, including a square check matrix.
    tiny = stim.DetectorErrorModel("error(0.01) D0 L0")
    decoder = CircuitBPOSD(tiny, {
        "max_bp_iters": 10,
        "bp_method": "minimum_sum",
        "osd_order": 0,
    })

    prediction = decoder.decode_batch(
        np.array([[1], [0]], dtype=np.uint8)
    )

    assert np.array_equal(
        prediction,
        np.array([[True], [False]], dtype=np.bool_),
    )

    print("DEM-converter and decoder self-tests passed.", flush=True)


# ---------- Fixed-shot benchmarking with checkpoints ----------

def benchmark_one(root, design, mode, basis, regime, p, args, settings):
    job = {
        "code": design.code.name,
        "mode": mode,
        "basis": basis,
        "regime": regime,
        "p": p,
        "decoder": settings,
    }
    job_id = digest_object(job)
    folder = root / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    stats_path = folder / "stats.json"

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats["shots"] >= args.shots:
            print(f"Already complete: {job}", flush=True)
            return
    else:
        stats = None

    start_setup = time.perf_counter()

    circuit, metadata = b.memory_circuit(
        design,
        mode,
        basis,
        noise_for(regime, p),
    )
    circuit = circuit.with_inlined_feedback()

    # Keep undecomposed multi-detector mechanisms.
    dem = circuit.detector_error_model(decompose_errors=False)
    decoder = CircuitBPOSD(dem, settings)

    if decoder.o.shape[0] != circuit.num_observables:
        raise RuntimeError("DEM/circuit observable-count mismatch")

    if stats is None:
        stats = {
            **job,
            "shots": 0,
            "block_failures": 0,
            "observable_failures": [0] * circuit.num_observables,
            "decode_seconds": 0.0,
            "sample_seconds": 0.0,
            "rounds": metadata["rounds"],
            "round_ticks": metadata["round_ticks"],
            "total_qubits": metadata["total_qubits"],
            "noise": asdict(noise_for(regime, p)),
            "fault_columns": int(decoder.h.shape[1]),
            "detectors": circuit.num_detectors,
            "observables": circuit.num_observables,
            "invisible_logical_mechanisms": decoder.invisible,
            "fault_distance_status": "not certified",
        }

    stats["latest_setup_seconds"] = time.perf_counter() - start_setup

    (folder / "circuit.stim").write_text(str(circuit), encoding="utf-8")
    (folder / "model.dem").write_text(str(dem), encoding="utf-8")
    write_json_atomic(folder / "metadata.json", metadata)
    write_json_atomic(stats_path, stats)

    print(
        f"\n{design.code.name} {mode} {basis} {regime} p={p:g}: "
        f"{decoder.h.shape[0]} detectors, "
        f"{decoder.h.shape[1]} fault columns, "
        f"{len(decoder.invisible)} invisible logical signatures",
        flush=True,
    )

    while stats["shots"] < args.shots:
        start_shot = stats["shots"]
        count = min(args.batch, args.shots - start_shot)

        # Each batch has a reproducible independent seed.
        # A resumed run starts from its last saved batch boundary.
        seed_text = f"{args.seed}:{job_id}:{start_shot}"
        seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
            byteorder="little",
        )

        start_sample = time.perf_counter()
        detectors, actual_observables = (
            circuit.compile_detector_sampler(seed=seed).sample(
                shots=count,
                separate_observables=True,
            )
        )
        stats["sample_seconds"] += (
            time.perf_counter() - start_sample
        )

        start_decode = time.perf_counter()
        predictions = decoder.decode_batch(detectors)
        stats["decode_seconds"] += (
            time.perf_counter() - start_decode
        )

        if predictions.shape != actual_observables.shape:
            raise RuntimeError("Prediction/sample shape mismatch")

        residual = np.logical_xor(predictions, actual_observables)

        stats["shots"] += count
        stats["block_failures"] += int(
            np.count_nonzero(np.any(residual, axis=1))
        )

        previous = np.array(
            stats["observable_failures"], dtype=np.int64
        )
        stats["observable_failures"] = (
            previous + residual.sum(axis=0, dtype=np.int64)
        ).tolist()

        if args.save_samples:
            np.savez_compressed(
                folder / f"batch_{start_shot:09d}.npz",
                detectors=detectors,
                actual_observables=actual_observables,
                predictions=predictions,
            )

        write_json_atomic(stats_path, stats)

        print(
            f"  {stats['shots']}/{args.shots} shots; "
            f"block failures={stats['block_failures']}; "
            f"estimate={stats['block_failures'] / stats['shots']:.5g}",
            flush=True,
        )


# ---------- Confidence intervals and output tables ----------

def exact_interval(failures, shots, confidence=0.95):
    result = binomtest(int(failures), int(shots))
    interval = result.proportion_ci(
        confidence_level=confidence,
        method="exact",
    )
    return float(interval.low), float(interval.high)


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    rows = []

    for path in sorted((root / "jobs").glob("*/stats.json")):
        stats = json.loads(path.read_text(encoding="utf-8"))
        shots = stats["shots"]

        if shots == 0:
            continue

        metrics = [("block", stats["block_failures"])]
        metrics.extend(
            (f"observable_{i}", failures)
            for i, failures in enumerate(stats["observable_failures"])
        )

        for metric, failures in metrics:
            low, high = exact_interval(failures, shots)

            rows.append({
                "code": stats["code"],
                "mode": stats["mode"],
                "basis": stats["basis"],
                "regime": stats["regime"],
                "p": stats["p"],
                "metric": metric,
                "shots": shots,
                "failures": failures,
                "ler": failures / shots,
                "ci95_low": low,
                "ci95_high": high,
                "rounds": stats["rounds"],
                "ticks_per_round": stats["round_ticks"][1],
                "observables": stats["observables"],
                "decode_seconds_per_shot": (
                    stats["decode_seconds"] / shots
                ),
                "fault_columns": stats["fault_columns"],
                "invisible_logical_signatures": len(
                    stats["invisible_logical_mechanisms"]
                ),
            })

    rows.sort(key=lambda row: (
        row["code"], row["basis"], row["regime"],
        row["p"], row["metric"], row["mode"],
    ))
    write_csv(root / "results.csv", rows)

    grouped = {}
    for row in rows:
        key = (
            row["code"], row["basis"], row["regime"],
            row["p"], row["metric"],
        )
        grouped.setdefault(key, {})[row["mode"]] = row

    comparisons = []

    for key, methods in sorted(grouped.items()):
        if "baseline" not in methods or "superdense" not in methods:
            continue

        baseline = methods["baseline"]
        superdense = methods["superdense"]

        # 97.5% intervals for each method give a conservative
        # >=95% joint rectangle for this one comparison.
        blo, bhi = exact_interval(
            baseline["failures"], baseline["shots"], 0.975
        )
        slo, shi = exact_interval(
            superdense["failures"], superdense["shots"], 0.975
        )

        difference = superdense["ler"] - baseline["ler"]
        difference_low = slo - bhi
        difference_high = shi - blo

        if difference_high < 0:
            conclusion = "superdense_lower"
        elif difference_low > 0:
            conclusion = "superdense_higher"
        else:
            conclusion = "inconclusive"

        ratio = (
            superdense["ler"] / baseline["ler"]
            if baseline["ler"] > 0 else None
        )
        ratio_low = slo / bhi if bhi > 0 else None
        ratio_high = shi / blo if blo > 0 else None

        code, basis, regime, p, metric = key

        comparisons.append({
            "code": code,
            "basis": basis,
            "regime": regime,
            "p": p,
            "metric": metric,
            "baseline_shots": baseline["shots"],
            "superdense_shots": superdense["shots"],
            "baseline_ler": baseline["ler"],
            "superdense_ler": superdense["ler"],
            "difference": difference,
            "difference95_low": difference_low,
            "difference95_high": difference_high,
            "ratio": ratio,
            "ratio95_low": ratio_low,
            "ratio95_high": ratio_high,
            "conclusion": conclusion,
        })

    write_csv(root / "comparisons.csv", comparisons)

    print("\nBlock-level comparisons:", flush=True)
    for row in comparisons:
        if row["metric"] == "block":
            print(
                f"{row['code']} {row['basis']} {row['regime']} "
                f"p={row['p']:g}: {row['conclusion']}; "
                f"baseline={row['baseline_ler']:.5g}, "
                f"superdense={row['superdense_ler']:.5g}",
                flush=True,
            )

    return rows, comparisons


# ---------- Plots ----------

def make_plots(root, rows, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combinations = sorted({
        (row["code"], row["basis"])
        for row in rows if row["metric"] == "block"
    })

    colors = {"baseline": "tab:blue", "superdense": "tab:orange"}

    for code, basis in combinations:
        regimes = sorted({
            row["regime"] for row in rows
            if row["code"] == code and row["basis"] == basis
        })

        fig, axes = plt.subplots(
            1, len(regimes),
            figsize=(5 * len(regimes), 4),
            squeeze=False,
        )

        for ax, regime in zip(axes.flat, regimes):
            for mode in ("baseline", "superdense"):
                selected = sorted(
                    [
                        row for row in rows
                        if row["code"] == code
                        and row["basis"] == basis
                        and row["regime"] == regime
                        and row["mode"] == mode
                        and row["metric"] == "block"
                    ],
                    key=lambda row: row["p"],
                )

                positive = [row for row in selected if row["failures"] > 0]

                if positive:
                    x = np.array([row["p"] for row in positive])
                    y = np.array([row["ler"] for row in positive])
                    lo = np.array([row["ci95_low"] for row in positive])
                    hi = np.array([row["ci95_high"] for row in positive])

                    ax.errorbar(
                        x, y,
                        yerr=np.maximum(
                            np.vstack([y - lo, hi - y]), 0
                        ),
                        fmt="o-",
                        capsize=3,
                        color=colors[mode],
                        label=mode,
                    )

                zero = [row for row in selected if row["failures"] == 0]
                if zero:
                    ax.scatter(
                        [row["p"] for row in zero],
                        [row["ci95_high"] for row in zero],
                        marker="v",
                        color=colors[mode],
                        label=f"{mode}: zero-count upper bound",
                    )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(regime)
            ax.set_xlabel("Reference physical error scale p")
            ax.set_ylabel("Block failure probability / memory experiment")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle(f"{code}, {basis}-memory, BP-OSD")
        fig.tight_layout()
        fig.savefig(root / f"{code}_{basis}_ler.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(
            1, len(regimes),
            figsize=(5 * len(regimes), 4),
            squeeze=False,
        )

        for ax, regime in zip(axes.flat, regimes):
            selected = sorted(
                [
                    row for row in comparisons
                    if row["code"] == code
                    and row["basis"] == basis
                    and row["regime"] == regime
                    and row["metric"] == "block"
                ],
                key=lambda row: row["p"],
            )

            if selected:
                x = np.array([row["p"] for row in selected])
                y = np.array([row["difference"] for row in selected])
                lo = np.array([
                    row["difference95_low"] for row in selected
                ])
                hi = np.array([
                    row["difference95_high"] for row in selected
                ])

                ax.errorbar(
                    x, y,
                    yerr=np.maximum(
                        np.vstack([y - lo, hi - y]), 0
                    ),
                    fmt="o-",
                    capsize=3,
                )

            ax.axhline(0, color="black", linestyle="--")
            ax.set_xscale("log")
            ax.set_title(regime)
            ax.set_xlabel("Reference physical error scale p")
            ax.set_ylabel("P(superdense) - P(baseline)")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{code}, {basis}-memory: negative means superdense helps"
        )
        fig.tight_layout()
        fig.savefig(root / f"{code}_{basis}_difference.png", dpi=180)
        plt.close(fig)


# ---------- Entry point ----------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default="part_c_output")
    parser.add_argument(
        "--codes", nargs="+", choices=list(CODE_FACTORIES),
        default=list(CODE_FACTORIES),
    )
    parser.add_argument(
        "--regimes", nargs="+", choices=list(REGIMES),
        default=["uniform"],
    )
    parser.add_argument("--bases", nargs="+", choices=["X", "Z"],
                        default=["X", "Z"])
    parser.add_argument("--p", nargs="+", type=float, default=[0.001])
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-bp-iters", type=int, default=30)
    parser.add_argument("--osd-order", type=int, default=0)
    parser.add_argument(
        "--bp-method",
        choices=["minimum_sum", "product_sum"],
        default="minimum_sum",
    )
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")

    args = parser.parse_args()

    if args.shots <= 0 or args.batch <= 0:
        parser.error("--shots and --batch must be positive")
    if args.max_bp_iters <= 0 or args.osd_order < 0:
        parser.error("Invalid decoder budget")
    if any(not 0 < p <= 0.1 for p in args.p):
        parser.error("This implementation requires 0 < p <= 0.1")

    self_test()
    if args.self_test_only:
        return

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    settings = {
        "max_bp_iters": args.max_bp_iters,
        "bp_method": args.bp_method,
        "osd_order": args.osd_order,
    }

    environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "stim": stim.__version__,
        "ldpc": version("ldpc"),
    }

    configuration = {
        "arguments": {
            key: value for key, value in vars(args).items()
            if key != "out"
        },
        "environment": environment,
        "part_b_sha256": hashlib.sha256(
            Path(b.__file__).read_bytes()
        ).hexdigest(),
        "part_c_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }

    config_path = root / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != configuration:
            raise SystemExit(
                "Output directory belongs to a different configuration. "
                "Use a new --out directory, or rerun the identical command."
            )
    else:
        write_json_atomic(config_path, configuration)

    for name in args.codes:
        code = CODE_FACTORIES[name]()
        design = b.design_for(code)

        for regime in args.regimes:
            for p in sorted(set(args.p)):
                for basis in args.bases:
                    for mode in ("baseline", "superdense"):
                        benchmark_one(
                            root, design, mode, basis,
                            regime, p, args, settings,
                        )

    rows, comparisons = summarize(root)
    make_plots(root, rows, comparisons)

    print(f"\nResults written to {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()