"""Fixed-key reachability under per-head integer query budgets."""

from fractions import Fraction
import math

import numpy as np
from scipy.optimize import linprog


MAX_COUNT = 4
SEARCH_STEPS = 16


def relaxation(denominator, difference, budgets):
    """Relax the normalized margin using one inverse denominator per head."""
    heads, width = denominator.shape
    rivals = difference.shape[1]
    size = heads * (width + 1) + 1
    inequalities = np.zeros((rivals, size))
    inequalities[:, -1] = 1
    equalities, upper, bounds = [], [0.] * rivals, []
    for h, total in enumerate(budgets):
        start = h * (width + 1)
        coords = slice(start, start + width)
        scale = denominator[h].max()
        if total == 0 or scale == 0:
            bounds.extend([(0, 0)] * (width + 1))
            continue
        bounds.extend([(0, None)] * (width + 1))
        inequalities[:rivals, coords] = -difference[h] / scale
        cap = np.zeros((width, size))
        cap[:, coords] = np.eye(width)
        cap[:, coords.stop] = -MAX_COUNT
        inequalities = np.vstack((inequalities, cap))
        upper.extend([0.] * width)
        budget = np.zeros(size)
        budget[coords] = 1
        budget[coords.stop] = -total
        equalities.append((budget, 0.))
        norm = np.zeros(size)
        norm[coords] = denominator[h] / scale
        silent_possible = MAX_COUNT * np.count_nonzero(denominator[h] == 0) >= total
        if silent_possible:
            inequalities = np.vstack((inequalities, norm))
            upper.append(1.)
        else:
            equalities.append((norm, 1.))
    bounds.append((-heads, heads))
    objective = np.zeros(size)
    objective[-1] = -1
    return linprog(objective, A_ub=inequalities, b_ub=upper,
                   A_eq=np.stack([row for row, _ in equalities]) if equalities else None,
                   b_eq=[rhs for _, rhs in equalities] if equalities else None, bounds=bounds, method="highs")


def round_counts(real, budget):
    """Project a candidate onto capped integers of the requested total."""
    real = np.clip(real, 0, MAX_COUNT)
    counts = np.floor(real).astype(np.int64)
    while counts.sum() != budget:
        if counts.sum() < budget:
            index = np.argmax(np.where(counts < MAX_COUNT, real - counts, -np.inf))
            counts[index] += 1
        else:
            index = np.argmax(np.where(counts > 0, counts - real, -np.inf))
            counts[index] -= 1
    return counts


def joint_margins(query, denominator, difference):
    normalizer = np.einsum("hd,hd->h", query, denominator)
    numerator = np.einsum("hd,hnd->hn", query, difference)
    return np.divide(numerator, normalizer[:, None], out=np.zeros_like(numerator),
                     where=normalizer[:, None] > 0)


def improve(seed, denominator, difference):
    """Greedy count transfers evaluated against the joint minimum margin."""
    q = seed.copy()
    for _ in range(SEARCH_STEPS):
        scores = joint_margins(q, denominator, difference)
        total = scores.sum(axis=0)
        best = total.min() + 1e-12
        move = None
        for h, row in enumerate(q):
            donors, recipients = np.where(row > 0)[0], np.where(row < MAX_COUNT)[0]
            if not len(donors) or not len(recipients):
                continue
            inc = difference[h][:, recipients].T
            dec = difference[h][:, donors].T
            counts = row[None, None, :] + np.eye(len(row), dtype=int)[recipients][None, :, :] \
                - np.eye(len(row), dtype=int)[donors][:, None, :]
            # Recompute positive denominators to preserve silent heads exactly.
            den = counts @ denominator[h]
            num = (difference[h] @ row)[None, None, :] + inc[None, :, :] - dec[:, None, :]
            ratios = np.divide(num, den[:, :, None], out=np.zeros_like(num), where=den[:, :, None] > 0)
            minima = (total - scores[h] + ratios).min(axis=-1)
            i, j = np.unravel_index(minima.argmax(), minima.shape)
            if minima[i, j] > best:
                best, move = minima[i, j], (h, donors[i], recipients[j])
        if move is None:
            break
        h, i, j = move
        q[h, i] -= 1
        q[h, j] += 1
    return q


def exact_geometry(keys, survival, source, rivals):
    """Store each head's weighted keys in a common integer unit."""
    geometry = []
    for k, factors in zip(keys, survival):
        dyadic = [float(s).as_integer_ratio() for s in factors]
        unit = max(d for _, d in dyadic)
        weighted = [[int(v) * n * (unit // d) for v in row]
                    for row, (n, d) in zip(k, dyadic)]
        denominator = list(map(sum, zip(*weighted)))
        difference = [[a - b for a, b in zip(weighted[source], weighted[j])] for j in rivals]
        geometry.append((denominator, difference))
    return geometry


def dot(a, b):
    return sum(int(x) * y for x, y in zip(a, b))


def exact_margin(query, geometry):
    margins = [Fraction(0)] * len(geometry[0][1])
    for row, (den, diff) in zip(query, geometry):
        normalizer = dot(row, den)
        if normalizer:
            margins = [m + Fraction(dot(row, d), normalizer) for m, d in zip(margins, diff)]
    return min(margins) / len(geometry)


def head_upper(denominator, numerator, budget, weight_sum):
    """Maximize a weighted ratio over the capped counts, with an exact fallback bound."""
    if budget == 0 or not any(denominator):
        return Fraction(0)
    silent = MAX_COUNT * denominator.count(0) >= budget
    ratio = Fraction(-1)
    for _ in range(64):
        order = sorted(range(len(denominator)),
                       key=lambda i: (numerator[i] * ratio.denominator
                                      - denominator[i] * weight_sum * ratio.numerator,
                                      denominator[i]), reverse=True)
        slots, remainder = divmod(budget, MAX_COUNT)
        n = MAX_COUNT * sum(numerator[i] for i in order[:slots])
        d = MAX_COUNT * sum(denominator[i] for i in order[:slots])
        if remainder:
            n += remainder * numerator[order[slots]]
            d += remainder * denominator[order[slots]]
        if n * ratio.denominator <= d * weight_sum * ratio.numerator:
            return max(Fraction(0), ratio) if silent else ratio
        ratio = Fraction(n, d * weight_sum)
    return Fraction(1)


def certificate(geometry, budgets, weights):
    """Any nonnegative integer competitor weights give a rigorous upper bound."""
    weight_sum = sum(weights)
    assert weight_sum > 0
    total = Fraction(0)
    for budget, (den, diff) in zip(budgets, geometry):
        numerator = [dot(weights, values) for values in zip(*diff)]
        total += head_upper(den, numerator, int(budget), weight_sum)
    return total / len(geometry)


def oracle(keys, survival, observed, source, rivals):
    """Return a checked witness, a certified nonpositive bound or an unresolved result."""
    if any((x < 0).any() or not np.isfinite(x).all() for x in (keys, survival, observed)):
        raise ValueError("oracle expects finite nonnegative keys, queries and survival")
    if ((observed > MAX_COUNT).any() or not np.equal(observed, np.floor(observed)).all()
            or (keys > MAX_COUNT).any() or not np.equal(keys, np.floor(keys)).all()):
        raise ValueError("oracle expects 0–4 count vectors")
    if not len(rivals) or source in rivals:
        raise ValueError("oracle requires at least one distinct rival")
    keys, observed = keys.astype(np.int64), observed.astype(np.int64)
    budgets = observed.sum(-1)
    weighted = keys * survival[:, :, None]
    denominator = weighted.sum(axis=1)
    difference = weighted[:, [source]] - weighted[:, rivals]
    geometry = exact_geometry(keys, survival, source, rivals)
    def wins_in_fp32(q):
        # Use the same rounded coefficients and FP64 share calculation as the diagnostic.
        coeff = (np.einsum("hd,htd->ht", q, keys) * survival).astype(np.float32).astype(np.float64)
        d = coeff.sum(-1, keepdims=True)
        share = np.divide(coeff, d, out=np.zeros_like(coeff), where=d != 0).mean(0)
        return share[source] > share[rivals].max()

    original = exact_margin(observed, geometry)
    if original > 0 and wins_in_fp32(observed):
        return dict(status="win", margin=float(original), query=observed.tolist())

    relaxed = relaxation(denominator, difference, budgets)
    combinations = [[1] * len(rivals)]
    seeds = [observed]
    if relaxed.success:
        rows = relaxed.x[:-1].reshape(len(budgets), -1)
        rounded = np.stack([round_counts(row[:-1] / row[-1], budget) if row[-1] > 0 else q
                            for row, budget, q in zip(rows, budgets, observed)])
        seeds.append(rounded)
        dual = np.maximum(0, -relaxed.ineqlin.marginals[:len(rivals)])
        if np.isfinite(dual).all() and dual.sum() > 0:
            combinations.insert(0, [int(round(x * (1 << 20))) for x in dual / dual.sum()])
    combinations.extend([int(i == j) for j in range(len(rivals))] for i in range(len(rivals)))
    bound = Fraction(1)
    for weights in combinations:
        upper = certificate(geometry, budgets, weights)
        bound = min(upper, bound)
        if upper <= 0:
            return dict(status="impossible", upper_bound=float(upper), bound_exact=str(upper),
                        competitor_weights=weights)

    best = original
    for seed in seeds:
        q = seed
        for attempt in range(2):
            if attempt:
                q = improve(seed, denominator, difference)
            assert np.all((0 <= q) & (q <= MAX_COUNT)) and np.array_equal(q.sum(-1), budgets)
            margin = exact_margin(q, geometry)
            if margin > 0 and wins_in_fp32(q):
                return dict(status="win", margin=float(margin), query=q.tolist())
            best = max(best, margin)
    return dict(status="unresolved", upper_bound=math.nextafter(float(bound), math.inf), best_margin=float(best))
