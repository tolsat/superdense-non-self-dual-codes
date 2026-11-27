from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import stim


# ---------- Binary linear algebra ----------

def rref(a, pivot_columns=None):
    a = np.array(a, dtype=np.uint8, copy=True) & 1
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {a.shape}")

    nrows, ncols = a.shape
    limit = ncols if pivot_columns is None else pivot_columns

    if not 0 <= limit <= ncols:
        raise ValueError("Invalid number of pivot columns")

    row = 0
    pivots = []

    for col in range(limit):
        if row >= nrows:
            break

        candidates = np.flatnonzero(a[row:, col])
        if len(candidates) == 0:
            continue

        # Use the FIRST available pivot row.
        p = row + int(candidates[0])
        a[[row,  p]] = a[[p,  row]]

        # Eliminate this column from all other ROWS.
        for k in range(nrows):
            if k != row and a[k,  col]:
                a[k] ^= a[row]

        pivots.append(col)
        row += 1

    return a, pivots


def row_basis(a):
    reduced, pivots = rref(a)
    return reduced[:len(pivots)]


def nullspace(a):
    reduced, pivots = rref(a)
    n = reduced.shape[1]
    free = [j for j in range(n) if j not in pivots]
    out = []

    for j in free:
        v = np.zeros(n, dtype=np.uint8)
        v[j] = 1

        for row, pivot in enumerate(pivots):
            v[pivot] = reduced[row,  j]

        out.append(v)

    return np.array(out, dtype=np.uint8).reshape(-1, n)


def quotient_basis(kernel, stabilizers):
    span = row_basis(stabilizers)
    n = stabilizers.shape[1]
    out = []

    for v in kernel:
        extended = row_basis(np.vstack([span,  v]))

        if len(extended) > len(span):
            out.append(v.copy())
            span = extended

    return np.array(out, dtype=np.uint8).reshape(-1, n)


def inverse_binary(a):
    a = np.array(a, dtype=np.uint8, copy=True) & 1

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Expected a square binary matrix")

    n = a.shape[1]
    augmented = np.hstack([
        a,
        np.eye(n, dtype=np.uint8),
    ])

    reduced, pivots = rref(augmented, pivot_columns=n)

    if pivots != list(range(n)):
        raise ValueError("Singular binary matrix")

    return reduced[:, n:]


def logical_operators(hx, hz):
    lx = quotient_basis(nullspace(hz), hx)
    lz = quotient_basis(nullspace(hx), hz)

    assert len(lx) == len(lz)

    pairing = (lx @ lz.T) & 1

    # Change the Z-logical basis so that X_i and Z_j
    # anticommute exactly when i == j.
    lz = (inverse_binary(pairing).T @ lz) & 1

    return lx, lz


def support(row):
    return np.flatnonzero(row).tolist()


# ---------- Code definitions ----------

@dataclass
class CSSCode:
    name: str
    distance: int
    hx: np.ndarray
    hz: np.ndarray
    lx: np.ndarray
    lz: np.ndarray
    data_xy: list
    x_xy: list
    z_xy: list
    left: set

    @property
    def n(self):
        return self.hx.shape[1]

    @property
    def k(self):
        return len(self.lx)


def check_code(code):
    hx, hz = code.hx, code.hz

    assert hx.ndim == 2, f"H_X is not 2D: {hx.shape}"
    assert hz.ndim == 2, f"H_Z is not 2D: {hz.shape}"

    assert hx.shape[1] == hz.shape[1]
    assert len(hx) == len(hz), (
        "This generator assumes equal X- and Z-check counts"
    )

    # CSS commutation condition.
    assert not np.any((hx @ hz.T) & 1), (
        "X and Z stabilizers do not commute"
    )

    rank_x = len(row_basis(hx))
    rank_z = len(row_basis(hz))
    k = code.n - rank_x - rank_z

    assert k == code.k, (
        f"Logical count mismatch: matrix calculation gives {k}, "
        f"but logical operators give {code.k}"
    )

    assert code.lx.shape == (k, code.n)
    assert code.lz.shape == (k, code.n)

    # Logical operators commute with opposite-type stabilizers.
    assert not np.any((hz @ code.lx.T) & 1)
    assert not np.any((hx @ code.lz.T) & 1)

    # Each logical X anticommutes with its paired logical Z.
    assert np.array_equal(
        (code.lx @ code.lz.T) & 1,
        np.eye(k, dtype=np.uint8),
    )


def surface_code(d):
    assert d >= 3 and d % 2 == 1
    n = d * d
    xs, zs = [], []

    def q(r, c):
        return r * d + c

    # Interior checks.
    for r in range(d - 1):
        for c in range(d - 1):
            s = [q(r, c), q(r, c + 1),
                 q(r + 1, c), q(r + 1, c + 1)]
            (xs if (r + c) % 2 == 0 else zs).append(s)

    # X boundaries: top and bottom.
    for c in range(d - 1):
        if c % 2 == 1:
            xs.append([q(0, c), q(0, c + 1)])
        if c % 2 == 0:
            xs.append([q(d - 1, c), q(d - 1, c + 1)])

    # Z boundaries: left and right.
    for r in range(d - 1):
        if r % 2 == 0:
            zs.append([q(r, 0), q(r + 1, 0)])
        if r % 2 == 1:
            zs.append([q(r, d - 1), q(r + 1, d - 1)])

    def matrix(checks):
        h = np.zeros((len(checks), n), dtype=np.uint8)
        for i, s in enumerate(checks):
            h[i,  s] = 1
        return h

    hx, hz = matrix(xs), matrix(zs)

    # Logical X: first column. Logical Z: top row.
    lx = np.zeros((1, n), dtype=np.uint8)
    lz = np.zeros((1, n), dtype=np.uint8)
    lx[0, [q(r, 0) for r in range(d)]] = 1
    lz[0, [q(0, c) for c in range(d)]] = 1

    xy = [(2 * c, 2 * r) for r in range(d) for c in range(d)]

    def center(s):
        return tuple(np.mean([xy[j] for j in s], axis=0).tolist())

    code = CSSCode(
        name=f"surface_d{d}",
        distance=d,
        hx=hx, hz=hz, lx=lx, lz=lz,
        data_xy=xy,
        x_xy=[center(s) for s in xs],
        z_xy=[center(s) for s in zs],
        left={q(r, c) for r in range(d)
              for c in range(d) if r % 2 == 0},
    )
    check_code(code)
    assert code.k == 1
    return code


def bb72_code():
    ell = m = 6
    size = ell * m

    def shift(n):
        s = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            s[i, (i + 1) % n] = 1
        return s

    x = np.kron(shift(ell), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(ell, dtype=np.uint8), shift(m))

    f = (np.linalg.matrix_power(x, 3)
         + y + np.linalg.matrix_power(y, 2)) & 1
    g = (np.linalg.matrix_power(y, 3)
         + x + np.linalg.matrix_power(x, 2)) & 1

    hx = np.hstack([f,  g])
    hz = np.hstack([g.T, f.T])
    lx, lz = logical_operators(hx, hz)

    cells = [(r, s) for r in range(ell) for s in range(m)]
    left_xy = [(4 * s, 4 * r) for r, s in cells]
    right_xy = [(4 * s + 2, 4 * r) for r, s in cells]

    code = CSSCode(
        name="bb72",
        distance=6,
        hx=hx, hz=hz, lx=lx, lz=lz,
        data_xy=left_xy + right_xy,
        x_xy=[(4 * s, 4 * r + 2) for r, s in cells],
        z_xy=[(4 * s + 2, 4 * r + 2) for r, s in cells],
        left=set(range(size)),
    )
    check_code(code)

    assert code.n == 72 and code.k == 12
    assert np.all(hx.sum(axis=1) == 6)
    assert np.all(hz.sum(axis=1) == 6)
    return code


# ---------- Pairing and hardware graph ----------

@dataclass
class Design:
    code: CSSCode
    pairs: list
    batches: list
    edges: set
    xy: list

    def ancillas(self, pair_index):
        a = self.code.n + 2 * pair_index
        return a, a + 1

    @property
    def total_qubits(self):
        return self.code.n + 2 * len(self.pairs)



def design_for(code):
    xs = [set(support(row)) for row in code.hx]
    zs = [set(support(row)) for row in code.hz]

    if len(xs) != len(zs):
        raise ValueError("Different numbers of X and Z checks")

    def score(i, j):
        separation = np.linalg.norm(
            np.array(code.x_xy[i]) - np.array(code.z_xy[j])
        )
        return len(xs[i] | zs[j]), float(separation), j

    # Only allow pairings satisfying the superdense parity condition.
    allowed = {}

    for i, sx in enumerate(xs):
        allowed[i] = []

        for j, sz in enumerate(zs):
            overlap = sx & sz

            left_parity = len(overlap & code.left) % 2
            right_parity = len(overlap - code.left) % 2

            if left_parity == 0 and right_parity == 0:
                allowed[i].append(j)

        allowed[i].sort(key=lambda j: score(i, j))

        if not allowed[i]:
            raise ValueError(
                f"No valid Z pairing for X stabilizer {i}. "
                "Try changing the left/right partition."
            )

    # Backtracking bipartite matching.
    # Select the X check with the fewest currently available partners.
    def find_matching(remaining, used, assignment):
        if not remaining:
            return assignment

        i = min(
            remaining,
            key=lambda i: sum(j not in used for j in allowed[i])
        )

        candidates = [j for j in allowed[i] if j not in used]

        if not candidates:
            return None

        for j in candidates:
            result = find_matching(
                [x for x in remaining if x != i],
                used | {j},
                {**assignment, i: j}
            )

            if result is not None:
                return result

        return None

    assignment = find_matching(
        list(range(len(xs))), set(), {}
    )

    if assignment is None:
        raise ValueError(
            "No valid one-to-one superdense pairing exists "
            "for this partition."
        )

    pairs = [(i, assignment[i]) for i in range(len(xs))]

    # Verify the matching.
    for i, j in pairs:
        overlap = xs[i] & zs[j]

        assert len(overlap & code.left) % 2 == 0
        assert len(overlap - code.left) % 2 == 0

    # Construct parallel batches.
    batches, occupied = [], []

    for p, (i, j) in enumerate(pairs):
        union = xs[i] | zs[j]

        for batch, used in zip(batches, occupied):
            if not (union & used):
                batch.append(p)
                used.update(union)
                break
        else:
            batches.append([p])
            occupied.append(set(union))

    # Construct hardware edges.
    edges = set()
    xy = list(code.data_xy)

    def add_edge(a, b):
        edges.add(tuple(sorted((a, b))))

    for p, (i, j) in enumerate(pairs):
        a = code.n + 2 * p
        b = a + 1

        xy.extend([code.x_xy[i], code.z_xy[j]])

        # Ordinary extraction connections.
        for q in xs[i]:
            add_edge(a, q)

        for q in zs[j]:
            add_edge(b, q)

        # Distributed extraction connections.
        for q in xs[i] | zs[j]:
            add_edge(a if q in code.left else b, q)

        # Bell-pair connection.
        add_edge(a, b)

    return Design(code, pairs, batches, edges, xy)



# ---------- Noise and scheduled circuit construction ----------

@dataclass
class Noise:
    p1: float = 0.0
    p2: float = 0.0
    pm: float = 0.0
    pr: float = 0.0
    pi: float = 0.0

    @classmethod
    def uniform(cls, p):
        return cls(p, p, p, p, p)


class Builder:
    def __init__(self, design, noise):
        self.design = design
        self.noise = noise
        self.circuit = stim.Circuit()
        self.nmeas = 0
        self.ticks = 0

        for q, xy in enumerate(design.xy):
            self.circuit.append("QUBIT_COORDS", [q], list(xy))

    def finish_layer(self, active):
        idle = sorted(set(range(self.design.total_qubits)) - set(active))
        if idle and self.noise.pi:
            self.circuit.append("DEPOLARIZE1", idle, self.noise.pi)
        self.circuit.append("TICK")
        self.ticks += 1

    def one(self, gate, qubits):
        qubits = list(qubits)
        if not qubits:
            return
        assert len(qubits) == len(set(qubits))

        self.circuit.append(gate, qubits)

        if gate in ("R", "RX"):
            if self.noise.pr:
                error = "X_ERROR" if gate == "R" else "Z_ERROR"
                self.circuit.append(error, qubits, self.noise.pr)
        elif gate == "H" and self.noise.p1:
            self.circuit.append("DEPOLARIZE1", qubits, self.noise.p1)

        self.finish_layer(qubits)

    def cx(self, gates):
        # Earliest-layer scheduling, preserving order on shared qubits.
        layers = []
        last_used = {}

        for control, target in gates:
            assert tuple(sorted((control, target))) in self.design.edges
            layer = max(
                last_used.get(control, -1),
                last_used.get(target, -1),
            ) + 1

            while len(layers) <= layer:
                layers.append([])

            layers[layer].append((control, target))
            last_used[control] = layer
            last_used[target] = layer

        for layer in layers:
            flat = [q for gate in layer for q in gate]
            assert len(flat) == len(set(flat)), "Parallel qubit conflict"

            self.circuit.append("CX", flat)
            if self.noise.p2:
                self.circuit.append("DEPOLARIZE2", flat, self.noise.p2)
            self.finish_layer(flat)

    def measure(self, gate, qubits):
        qubits = list(qubits)
        assert len(qubits) == len(set(qubits))

        self.circuit.append(gate, qubits, self.noise.pm)
        records = list(range(self.nmeas, self.nmeas + len(qubits)))
        self.nmeas += len(qubits)
        self.finish_layer(qubits)
        return records

    def rec(self, absolute_index):
        offset = absolute_index - self.nmeas
        assert offset < 0
        return stim.target_rec(offset)

    def detector(self, records, coordinates):
        self.circuit.append(
            "DETECTOR",
            [self.rec(r) for r in records],
            coordinates,
        )

    def feedback_x(self, record, qubits):
        # Ideal Pauli-frame update: no physical gate or time cost.
        for q in qubits:
            self.circuit.append("CX", [self.rec(record), q])


def extract_batch(builder, batch, mode):
    design = builder.design
    code = design.code
    anc = [q for p in batch for q in design.ancillas(p)]
    a_qubits = [design.ancillas(p)[0] for p in batch]

    builder.one("R", anc)
    builder.one("H", a_qubits)

    if mode == "superdense":
        builder.cx([design.ancillas(p) for p in batch])

    z_gates, x_gates = [], []

    for p in batch:
        i, j = design.pairs[p]
        a, b = design.ancillas(p)

        for q in support(code.hz[j]):
            target = (
                b if mode == "baseline"
                else (a if q in code.left else b)
            )
            z_gates.append((q, target))

        for q in support(code.hx[i]):
            control = (
                a if mode == "baseline"
                else (a if q in code.left else b)
            )
            x_gates.append((control, q))

    # Keep the block order proved in Part A.
    builder.cx(z_gates)
    builder.cx(x_gates)

    if mode == "superdense":
        builder.cx([design.ancillas(p) for p in batch])

    builder.one("H", a_qubits)
    records = builder.measure("M", anc)

    outcomes = {}
    for position, p in enumerate(batch):
        i, j = design.pairs[p]
        u = records[2 * position]
        v = records[2 * position + 1]

        outcomes[("X", i)] = u
        outcomes[("Z", j)] = v

        if mode == "superdense":
            xr = [q for q in support(code.hx[i]) if q not in code.left]
            builder.feedback_x(v, xr)

    return outcomes


def memory_circuit(design, mode, basis, noise, injected=None):
    assert mode in ("baseline", "superdense")
    assert basis in ("X", "Z")
    code = design.code
    builder = Builder(design, noise)

    builder.one("R" if basis == "Z" else "RX", range(code.n))
    previous = None
    round_ticks = []

    for t in range(code.distance):
        start = builder.ticks

        # Optional ideal inserted fault for validation.
        if injected is not None:
            injection_round, pauli, q = injected

            if injection_round == t:
                builder.circuit.append(pauli, [q])

        current = {}
        for batch in design.batches:
            current.update(extract_batch(builder, batch, mode))

        for kind, matrix in (("X", code.hx), ("Z", code.hz)):
            for i in range(len(matrix)):
                key = (kind, i)
                if previous is not None:
                    builder.detector(
                        [current[key], previous[key]],
                        [i, 0 if kind == "X" else 1, t],
                    )
                elif kind == basis:
                    builder.detector(
                        [current[key]],
                        [i, 0 if kind == "X" else 1, t],
                    )

        previous = current
        round_ticks.append(builder.ticks - start)

    final = builder.measure(
        "M" if basis == "Z" else "MX",
        range(code.n),
    )

    closing_checks = code.hz if basis == "Z" else code.hx
    for i, row in enumerate(closing_checks):
        builder.detector(
            [previous[(basis, i)]] + [final[q] for q in support(row)],
            [i, 0 if basis == "X" else 1, code.distance],
        )

    logicals = code.lz if basis == "Z" else code.lx
    for i, row in enumerate(logicals):
        builder.circuit.append(
            "OBSERVABLE_INCLUDE",
            [builder.rec(final[q]) for q in support(row)],
            i,
        )

    metadata = {
        "code": code.name,
        "mode": mode,
        "basis": basis,
        "rounds": code.distance,
        "data_qubits": code.n,
        "total_qubits": design.total_qubits,
        "logical_qubits": code.k,
        "pair_batches": len(design.batches),
        "round_ticks": round_ticks,
        "noise": asdict(noise),
        "fault_distance_status": "not certified",
    }
    return builder.circuit, metadata


# ---------- Validation ----------

def validate_memory(design, mode, basis, full_tests=False):
    ideal, _ = memory_circuit(design, mode, basis, Noise())

    # Strict conversion rejects nondeterministic declared detectors.
    ideal.detector_error_model()

    detectors, observables = ideal.compile_detector_sampler(
        seed=1
    ).sample(shots=256, separate_observables=True)

    assert not detectors.any(), "Noiseless detector failure"
    assert not observables.any(), "Noiseless observable failure"

    # Compare faulty measurements against the IDEAL reference circuit.
    # A sampler compiled from the faulty circuit alone would absorb a
    # deterministic inserted fault into its own reference trajectory.
    converter = ideal.compile_m2d_converter()
    code = design.code

    qubits = (
        range(code.n) if full_tests
        else sorted({0, code.n // 2, code.n - 1})
    )

    first_round_count = len(code.hz if basis == "Z" else code.hx)
    mx, mz = len(code.hx), len(code.hz)

    for q in qubits:
        for pauli in ("X", "Y", "Z"):
            faulty, _ = memory_circuit(
                design, mode, basis, Noise(),
                injected=(1, pauli, q),
            )
            measurements = faulty.compile_sampler(seed=2).sample(shots=16)
            det, obs = converter.convert(
                measurements=measurements,
                separate_observables=True,
            )

            expected_x = (
                code.hx[:, q] if pauli in ("Z", "Y")
                else np.zeros(mx, dtype=np.uint8)
            )
            expected_z = (
                code.hz[:, q] if pauli in ("X", "Y")
                else np.zeros(mz, dtype=np.uint8)
            )

            start = first_round_count
            assert np.all(det[:, start:start + mx] == expected_x)
            actual_z = det[:, start + mx:start + mx + mz]

            if not np.all(actual_z == expected_z[None, :]):
                print("\nZ syndrome validation failed")
                print("Code:", code.name)
                print("Mode:", mode)
                print("Basis:", basis)
                print("Qubit:", q)
                print("Pauli:", pauli)

                print("Expected:", expected_z.astype(int))
                print("Actual:  ", actual_z[0].astype(int))

                raise AssertionError("Incorrect Z syndrome")

            logicals = code.lz if basis == "Z" else code.lx
            flips_memory = (
                pauli in ("X", "Y") if basis == "Z"
                else pauli in ("Z", "Y")
            )
            expected_obs = (
                logicals[:, q] if flips_memory
                else np.zeros(code.k, dtype=np.uint8)
            )
            assert np.all(obs == expected_obs)

    return ideal


# ---------- Output ----------

def save_layout(design, folder):
    data = {
        "coordinates": [
            {"qubit": q, "xy": list(xy)}
            for q, xy in enumerate(design.xy)
        ],
        "native_edges": [list(e) for e in sorted(design.edges)],
        "pairs": [
            {"pair": p, "x_check": i, "z_check": j,
             "ancillas": list(design.ancillas(p))}
            for p, (i, j) in enumerate(design.pairs)
        ],
        "batches": design.batches,
        "assumption": "Both CNOT directions available on every native edge",
    }
    (folder / "hardware.json").write_text(json.dumps(data, indent=2))

    # Simple hardware graph visualization.
    points = np.array(design.xy, dtype=float)
    lo = points.min(axis=0)
    points = (points - lo) * 24 + 30
    width, height = points.max(axis=0) + 30
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">'
    ]
    for a, b in sorted(design.edges):
        x1, y1 = points[a]
        x2, y2 = points[b]
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#bbb" stroke-width="1"/>'
        )
    for q, (x, y) in enumerate(points):
        color = "#444" if q < design.code.n else (
            "#d45" if (q - design.code.n) % 2 == 0 else "#36c"
        )
        svg.append(
            f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>'
        )
        svg.append(
            f'<text x="{x + 6}" y="{y - 6}" font-size="9">{q}</text>'
        )
    svg.append("</svg>")
    (folder / "hardware.svg").write_text("\n".join(svg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="part_b_output")
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--full-tests", action="store_true")
    args = parser.parse_args()
    assert 0 <= args.p <= 1

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "environment.json").write_text(json.dumps({
        "python": sys.version,
        "numpy": np.__version__,
        "stim": stim.__version__,
        "arguments": vars(args),
    }, indent=2))

    codes = [surface_code(3), surface_code(5), bb72_code()]

    for code in codes:
        design = design_for(code)
        folder = root / code.name
        folder.mkdir(exist_ok=True)
        save_layout(design, folder)

        np.savez(
            folder / "code_matrices.npz",
            hx=code.hx, hz=code.hz,
            lx=code.lx, lz=code.lz,
        )

        for mode in ("baseline", "superdense"):
            for basis in ("X", "Z"):
                name = f"{basis}_{mode}"
                print(
                    f"Validating {code.name}, mode={mode}, basis={basis}",
                    flush=True
                )
                ideal = validate_memory(
                    design, mode, basis, args.full_tests
                )
                (folder / f"{name}_ideal.stim").write_text(str(ideal))

                noisy, metadata = memory_circuit(
                    design, mode, basis, Noise.uniform(args.p)
                )

                # Auditable circuit with explicit frame updates.
                (folder / f"{name}_feedback.stim").write_text(str(noisy))

                # Decoder-facing circuit with feedback inlined.
                decoder_circuit = noisy.with_inlined_feedback()
                dem = decoder_circuit.detector_error_model(
                    decompose_errors=False
                )

                (folder / f"{name}.stim").write_text(str(decoder_circuit))
                (folder / f"{name}.dem").write_text(str(dem))
                (folder / f"{name}.json").write_text(
                    json.dumps(metadata, indent=2)
                )

                det, obs = decoder_circuit.compile_detector_sampler(
                    seed=3
                ).sample(
                    shots=args.shots,
                    separate_observables=True,
                )
                np.savez(
                    folder / f"{name}_samples.npz",
                    detectors=det,
                    observables=obs,
                )

                # Timeline visualization of the noiseless circuit.
                (folder / f"{name}_timeline.svg").write_text(
                    str(ideal.diagram("timeline-svg"))
                )

                print(
                    code.name, name,
                    "qubits=", decoder_circuit.num_qubits,
                    "detectors=", decoder_circuit.num_detectors,
                    "observables=", decoder_circuit.num_observables,
                    "round_ticks=", metadata["round_ticks"],
                    "VALIDATION PASSED",
                )


if __name__ == "__main__":
    main()