"""Common candidate sampler, a count heuristic, and an online NumPy MLP.

The learner gets only functional-coverage novelty after simulation. It cannot
see the mutation identity, oracle mismatches, or candidate outputs beforehand.
"""
from collections import Counter

import numpy as np

from .hardware import Case, MASK, Op

EDGES = (0, 1, 2, 15, 16, 31, 32, 33, 63, 64, 0x7FFFFFFF, 0x80000000, MASK)


def value_class(value):
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    if value == MASK:
        return "all_ones"
    if value == 0x80000000:
        return "signed_min"
    if value == 0x7FFFFFFF:
        return "signed_max"
    if value & (value - 1) == 0:
        return "power_two"
    return "negative" if value >> 31 else "positive"


CLASSES = ("zero", "one", "all_ones", "signed_min", "signed_max", "power_two", "negative", "positive")


def input_bins(case):
    op = case.op.name
    bins = {f"{op}:a:{value_class(case.a)}", f"{op}:b:{value_class(case.b)}"}
    if case.op in (Op.SLL, Op.SRL, Op.SRA):
        bins.add(f"{op}:shift:{case.b & 31}")
    return bins


def coverage_bins(case, result):
    bins = input_bins(case)
    bins.add(f"{case.op.name}:output:{value_class(result)}")
    bins.add(f"{case.op.name}:signs:{case.a >> 31}{case.b >> 31}{result >> 31}")
    return bins


def candidates(rng, size):
    def operand():
        # All strategies use exactly the same boundary-biased sampler.
        if rng.random() < 0.35:
            return int(rng.choice(EDGES))
        return int(rng.integers(0, 1 << 32, dtype=np.uint64))
    return [Case(Op(int(rng.integers(len(Op)))), operand(), operand()) for _ in range(size)]


def features(case):
    values = [float(case.op == op) for op in Op]
    for value in (case.a, case.b):
        values += [float(value_class(value) == c) for c in CLASSES]
        values += [value / MASK, (value & 31) / 31, value.bit_count() / 32]
    values += [float(case.a == case.b), float(case.a < case.b)]
    return np.asarray(values, dtype=np.float64)


class NeuralSelector:
    """One hidden layer; BCE on 'this execution adds a coverage bin'."""

    def __init__(self, seed, dimensions, hidden=24):
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.normal(0, 1 / np.sqrt(dimensions), (dimensions, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = self.rng.normal(0, 1 / np.sqrt(hidden), hidden)
        self.b2 = 0.0
        self.x, self.y = [], []

    def predict(self, x):
        hidden = np.tanh(x @ self.w1 + self.b1)
        return 1 / (1 + np.exp(-np.clip(hidden @ self.w2 + self.b2, -30, 30)))

    def observe(self, x, novelty):
        self.x.append(x)
        self.y.append(float(novelty))
        if len(self.x) < 32 or len(self.x) % 16:
            return
        # Uniform replay; historical novelty labels are nonstationary by design.
        # No mutation-specific positive examples are supplied.
        for _ in range(8):
            idx = self.rng.choice(len(self.x), min(64, len(self.x)), replace=False)
            x_batch = np.asarray([self.x[i] for i in idx])
            y_batch = np.asarray([self.y[i] for i in idx])
            hidden = np.tanh(x_batch @ self.w1 + self.b1)
            prediction = 1 / (1 + np.exp(-np.clip(hidden @ self.w2 + self.b2, -30, 30)))
            dz = (prediction - y_batch) / len(idx)
            dw2 = hidden.T @ dz
            dh = dz[:, None] * self.w2 * (1 - hidden * hidden)
            self.w2 -= 0.05 * dw2
            self.b2 -= 0.05 * dz.sum()
            self.w1 -= 0.05 * (x_batch.T @ dh)
            self.b1 -= 0.05 * dh.sum(axis=0)


class Selector:
    def __init__(self, strategy, seed):
        self.strategy = strategy
        self.rng = np.random.default_rng(seed + 100000)
        self.counts = Counter()
        self.model = NeuralSelector(seed + 200000, len(features(Case(Op.ADD, 0, 0))))
        self.steps = 0

    def choose(self, pool):
        if self.strategy == "random" or self.steps < 32 or self.rng.random() < 0.15:
            return pool[0]
        if self.strategy == "coverage":
            scores = [sum(1 / (1 + self.counts[b]) for b in input_bins(c)) for c in pool]
        else:
            scores = self.model.predict(np.stack([features(c) for c in pool]))
        return pool[int(np.argmax(scores))]

    def observe(self, case, novel):
        self.counts.update(input_bins(case))
        if self.strategy == "neural":
            self.model.observe(features(case), novel)
        self.steps += 1
