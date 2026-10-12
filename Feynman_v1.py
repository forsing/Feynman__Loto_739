#!/usr/bin/env python3

"""Feynman V1: deterministicka adaptacija sume amplituda preko putanja."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

WINDOW, PATHS, STARTS = 96, 24, 12
ALPHAS = (.01, .1, 1.)
BANDS = np.array((.5, 1., 2.))
DIM = 30


def load(path):
    rows = []

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != 7
                or values != sorted(values)
                or len(set(values)) != 7
                or not all(1 <= v <= 39 for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna kombinacija 7/39."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1) - 1.


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def support(history):
    tree = cKDTree(gaps(history) / 5.)

    distance = tree.query(
        gaps(history) / 5., k=2, p=1
    )[0][:, 1]

    return tree, max(
        float(np.quantile(distance, .75)),
        1e-12
    )


def allowed(rows, tree, radius):
    distance = tree.query(
        gaps(rows) / 5., p=1
    )[0]

    return distance <= radius + 1e-10


def context(g):
    # Samo prethodno ostvareni prelazi;
    # kandidat nije medju putanjama.
    changes = np.diff(g, axis=0) / 5.

    return (
        g[-1],
        changes[-1],
        changes[-PATHS:]
    )


def path_features(delta, memory, paths):
    # Dve deonice u prostoru PRELAZA:
    # memory -> path -> delta.
    first = np.sum(
        np.abs(paths - memory), axis=1
    )

    weight = 1. / (1. + first)
    weight /= weight.sum()

    second = np.sum(
        np.abs(delta[:, None, :] - paths),
        axis=2
    )

    action = first[None, :] + second

    phase = (
        action[:, :, None]
        * BANDS[None, None, :]
    )

    amplitude = np.sum(
        weight[None, :, None] * np.exp(1j * phase),
        axis=1
    )

    coherent = np.abs(amplitude) ** 2

    # Nekohorentna kontrola: uklonjeni unakrsni
    # clanovi interferencije.
    incoherent = float(weight @ weight)

    return coherent, coherent - incoherent


def features(g, ctx, interference=True):
    reference, memory, paths = ctx

    delta = (
        np.atleast_2d(g) - reference
    ) / 5.

    coherent, cross = path_features(
        delta, memory, paths
    )

    if not interference:
        # Sabiranje intenziteta istih putanja
        # umesto amplituda.
        first = np.sum(
            np.abs(paths - memory), axis=1
        )

        weight = 1. / (1. + first)
        weight /= weight.sum()

        coherent = np.full_like(
            coherent, weight @ weight
        )

        cross = np.zeros_like(cross)

    return np.column_stack((
        delta,
        np.abs(delta),
        delta * memory,
        coherent,
        cross
    ))


def prepare(history, interference=True):
    needed = WINDOW + PATHS + 1

    if len(history) < needed:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{needed} prethodnih redova."
        )

    h = history[-needed:]
    g = gaps(h)

    gram = np.zeros((DIM, DIM))
    rhs = np.zeros(DIM)

    for t in range(PATHS + 1, len(h)):
        ctx = context(g[:t])

        actual = features(
            g[t:t + 1], ctx, interference
        )

        other = features(
            gaps(replacements(h[t])),
            ctx,
            interference
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast / len(contrast)
        )

        rhs += contrast.mean(axis=0)

    gram /= WINDOW
    rhs /= WINDOW

    # RMS je numericka skala osobina,
    # nije ciljna raspodela brojeva.
    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        context(g),
        h[-WINDOW:],
        interference
    )


def fit(prepared, alpha):
    gram, rhs, scale, ctx, _, interference = prepared

    coefficient = (
        np.linalg.solve(
            gram + alpha * np.eye(DIM),
            rhs
        ) / scale
    )

    return coefficient, ctx, interference


def scores(rows, model):
    coefficient, ctx, interference = model

    result = (
        features(
            gaps(rows), ctx, interference
        ) @ coefficient
    )

    if not np.all(np.isfinite(result)):
        raise ArithmeticError("Neispravan skor.")

    return result


def search(history, model):
    tree, radius = support(history)

    seeds = np.unique(history, axis=0)

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )[:STARTS]

    current = seeds[order].copy()
    iterations = 0

    while True:
        moved = False

        for j, row in enumerate(current):
            candidates = replacements(row)

            candidates = candidates[
                allowed(candidates, tree, radius)
            ]

            if not len(candidates):
                continue

            values = scores(candidates, model)
            best = int(np.argmax(values))

            if (
                values[best]
                > scores(row, model)[0] + 1e-10
            ):
                current[j] = candidates[best]
                moved = True

        if not moved:
            break

        iterations += 1

    values = scores(current, model)
    best = int(np.argmax(values))

    return (
        tuple(map(int, current[best])),
        float(values[best]),
        iterations
    )


def predict(prepared, alpha):
    return search(
        prepared[4],
        fit(prepared, alpha)
    )


def hits(prediction, actual):
    return len(
        set(prediction).intersection(actual)
    )


def run_file(path, validation=48, test=64):
    history = load(path)
    split = len(history) - test

    if (
        validation < 16
        or test < 16
        or split - validation < WINDOW + PATHS + 1
    ):
        raise ValueError(
            "Nedovoljno podataka ili prekratka provera."
        )

    labels = {
        True: "FEYNMAN: INTERFERENCIJA",
        False: "Kontrola: bez interferencije"
    }

    chosen = {}

    # Parametri se biraju pre odvojenog testa.
    # Oba modela imaju sopstveni izbor parametra.
    for mode in (True, False):
        totals = np.zeros(
            len(ALPHAS), dtype=int
        )

        for t in range(split - validation, split):
            p = prepare(history[:t], mode)

            for j, alpha in enumerate(ALPHAS):
                totals[j] += hits(
                    predict(p, alpha)[0],
                    history[t]
                )

        chosen[mode] = ALPHAS[
            int(np.argmax(totals))
        ]

    result = {
        True: [],
        False: [],
        "previous": []
    }

    # Parametri su zamrznuti.
    # Svaki korak koristi samo prethodna kola.
    for t in range(split, len(history)):
        for mode in (True, False):
            p = prepare(history[:t], mode)

            prediction = predict(
                p, chosen[mode]
            )[0]

            result[mode].append(
                hits(prediction, history[t])
            )

        result["previous"].append(
            hits(history[t - 1], history[t])
        )

    print("\nCSV:", path)

    print(
        "Hronologija: CSV mora biti od najstarijeg "
        "ka najnovijem kolu."
    )

    print(
        f"Validacija: {validation}; "
        f"odvojeni test: {test}"
    )

    for mode in (True, False):
        values = result[mode]

        print(
            f"{labels[mode]}: "
            f"alpha={chosen[mode]:g}, "
            f"prosek={np.mean(values):.4f}, "
            f"pogoci 0..7="
            f"{np.bincount(values, minlength=8).tolist()}"
        )

    print(
        "Kontrola: prethodno izvuceno kolo:",
        f"{np.mean(result['previous']):.4f}"
    )

    p = prepare(history)

    prediction, score, steps = predict(
        p, chosen[True]
    )

    replay = bool(
        np.any(
            np.all(history == prediction, axis=1)
        )
    )

    print(
        f"SLEDECI RED {len(history) + 1}:",
        ",".join(map(str, prediction))
    )

    print(
        f"Skor={score:.10f}; "
        f"koraci={steps}; "
        f"istorijsko ponavljanje={replay}"
    )

    print(
        "Skor nije verovatnoca. Pretraga: "
        "istorijska polazista i dozvoljene"
    )

    print(
        "zamene jednog broja; lokalni optimum, "
        "bez tvrdnje o globalnom."
    )

    return prediction


def self_test():
    base = np.array((
        (1, 2, 3, 4, 5, 6, 7),
        (4, 12, 18, 28, 35, 36, 37),
        (33, 34, 35, 36, 37, 38, 39),
        (2, 7, 12, 18, 24, 30, 39)
    ))

    g = gaps(base)

    assert np.all(g.sum(axis=1) == 32)

    assert np.array_equal(
        np.cumsum(g[:, :-1] + 1, axis=1),
        base
    )

    h = np.tile(base, (40, 1))

    for mode in (True, False):
        p = prepare(h, mode)
        model = fit(p, .1)

        scaled = model[0] * p[2]

        assert np.allclose(
            (p[0] + .1 * np.eye(DIM)) @ scaled,
            p[1]
        )

        prediction, score, _ = predict(p, .1)

        assert prediction == predict(
            prepare(h, mode), .1
        )[0]

        assert (
            len(set(prediction)) == 7
            and 1 <= min(prediction)
            <= max(prediction) <= 39
        )

        tree, radius = support(p[4])

        other = replacements(
            np.array(prediction)
        )

        other = other[
            allowed(other, tree, radius)
        ]

        assert (
            scores(other, model).max(initial=-np.inf)
            <= score + 1e-9
        )

    delta = np.array((
        (0.,) * 8,
        (1., -1., 0., 0., 0., 0., 0., 0.)
    ))

    paths = np.diff(g, axis=0) / 5.

    coherent, cross = path_features(
        delta, paths[-1], paths
    )

    first = np.abs(
        paths - paths[-1]
    ).sum(axis=1)

    w = 1 / (1 + first)
    w /= w.sum()

    action = (
        first[None, :]
        + np.abs(
            delta[:, None, :] - paths
        ).sum(axis=2)
    )

    # Nezavisna provera:
    # |sum amplituda|^2 = suma svih parnih clanova.
    direct = np.empty_like(coherent)

    for n in range(len(delta)):
        for k, band in enumerate(BANDS):
            direct[n, k] = sum(
                w[a] * w[b] * np.cos(
                    band * (
                        action[n, a] - action[n, b]
                    )
                )
                for a in range(len(w))
                for b in range(len(w))
            )

    assert np.allclose(coherent, direct)
    assert np.allclose(cross, direct - w @ w)
    assert np.all(coherent <= 1 + 1e-12)

    future = h.copy()
    future[-1] = base[1]

    assert predict(
        prepare(h[:-1]), .1
    )[0] == predict(
        prepare(future[:-1]), .1
    )[0]

    print(
        "Self-test OK: razmaci, interferencija, "
        "ucenje, ponovljivost i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Podrazumevano sva tri CSV fajla."
    )

    parser.add_argument(
        "--validation",
        type=int,
        default=48
    )

    parser.add_argument(
        "--test",
        type=int,
        default=64
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    for path in args.csv or DEFAULTS:
        run_file(
            path,
            args.validation,
            args.test
        )


if __name__ == "__main__":
    main()



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
FEYNMAN: INTERFERENCIJA: alpha=0.1, prosek=1.4219, pogoci 0..7=[11, 25, 19, 8, 1, 0, 0, 0]
Kontrola: bez interferencije: alpha=0.1, prosek=1.1250, pogoci 0..7=[15, 29, 17, 3, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2188
SLEDECI RED 4699: 1,2,6,7,16,35,36
Skor=1.2174473380; koraci=15; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.



CSV: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
FEYNMAN: INTERFERENCIJA: alpha=0.1, prosek=1.1406, pogoci 0..7=[17, 28, 12, 7, 0, 0, 0, 0]
Kontrola: bez interferencije: alpha=0.1, prosek=1.3906, pogoci 0..7=[14, 24, 15, 9, 2, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2969
SLEDECI RED 2972: 1,3,5,7,8,10,39
Skor=1.3920426829; koraci=11; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.



CSV: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64
FEYNMAN: INTERFERENCIJA: alpha=1, prosek=1.2344, pogoci 0..7=[16, 23, 19, 6, 0, 0, 0, 0]
Kontrola: bez interferencije: alpha=1, prosek=1.2500, pogoci 0..7=[14, 26, 18, 6, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: 1.2969
SLEDECI RED 1728: 1,2,6,7,11,14,15
Skor=0.7159047675; koraci=9; istorijsko ponavljanje=False
Skor nije verovatnoca. Pretraga: istorijska polazista i dozvoljene
zamene jednog broja; lokalni optimum, bez tvrdnje o globalnom.
"""



"""
Biram sumu amplituda preko putanja i interferenciju iz Feynmanovog rada iz 1948. 
Za Loto ću je prilagoditi tako da istorijski prelazi između celih kombinacija doprinose zajedničkom skoru kandidata; brojevi se neće prosečavati. 
To je jedna obrazložena ideja za ovaj model, a njenu korisnost proveravam na izdvojenim narednim kolima. 

Provere formule interferencije, ponovljivosti i odvojenog učenja za sva tri fajla su prošle. 
Test na poslednja 64 kola, uz kontrolni model bez interferencije; pokazace da li baš ova Feynmanova ideja doprinosi rezultatu.

Kandidat dobija skor prema zajedničkom doprinosu istorijskih prelaza između celih kombinacija. 
Konkretna definicija putanja i njihove akcije prilagođena je Lotu. 
Interferencija je pomogla na zbirnom fajlu, ali prednost nije dosledna na sva tri.

CSV	        Prosek pogodaka	Bez interferencije	Sledeća kombinacija
Zbirni 4698	1,4219	        1,1250	            1,2,6,7,16,35,36
Loto 2971	1,1406	        1,3906	            1,3,5,7,8,10,39
Plus 1727	1,2344	        1,2500	            1,2,6,7,11,14,15
"""
