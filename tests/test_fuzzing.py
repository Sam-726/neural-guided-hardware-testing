import unittest

import numpy as np
from amaranth.sim import Simulator

from hwfuzz.experiment import reduce_case, replay, run_trial
from hwfuzz.guidance import EDGES, NeuralSelector, candidates, coverage_bins, features
from hwfuzz.hardware import ALU, Case, MASK, MUTANTS, Op, oracle


class HardwareTests(unittest.TestCase):
    def test_known_rv32i_semantics(self):
        vectors = [
            (Op.ADD, MASK, 1, 0), (Op.SUB, 0, 1, MASK),
            (Op.SRA, 0x80000000, 1, 0xC0000000),
            (Op.SRL, 0x80000000, 1, 0x40000000),
            (Op.SLL, 1, 32, 1), (Op.SLL, 1, 63, 0x80000000),
            (Op.SLT, MASK, 0, 1), (Op.SLTU, MASK, 0, 0),
            (Op.AND, 0xAA, 0x0F, 0x0A), (Op.OR, 0xAA, 0x0F, 0xAF),
            (Op.XOR, 0xAA, 0x0F, 0xA5),
        ]
        for op, a, b, expected in vectors:
            with self.subTest(op=op, a=a, b=b):
                self.assertEqual(oracle(Case(op, a, b)), expected)

    def test_hdl_against_oracle_boundary_cross_product_and_random(self):
        dut = ALU()
        cases = [Case(op, a, b) for op in Op for a in EDGES for b in EDGES]
        cases += candidates(np.random.default_rng(2026), 1000)
        async def bench(ctx):
            for case in cases:
                ctx.set(dut.op, int(case.op))
                ctx.set(dut.a, case.a)
                ctx.set(dut.b, case.b)
                await ctx.delay(1e-9)
                self.assertEqual(ctx.get(dut.y), oracle(case), case)
        sim = Simulator(dut)
        sim.add_testbench(bench)
        sim.run()

    def test_mutations_and_reduction_replay(self):
        examples = dict(zip(MUTANTS, [Case(Op.SRA, MASK, 31), Case(Op.SLT, MASK, 0),
                                     Case(Op.SLL, 1, 32), Case(Op.ADD, MASK, MASK)]))
        entries = []
        for mutant, original in examples.items():
            smaller = reduce_case(mutant, original)
            self.assertLessEqual(smaller.a.bit_count() + smaller.b.bit_count(),
                                 original.a.bit_count() + original.b.bit_count())
            entries.append({'mutant': mutant, **smaller.to_dict()})
        self.assertEqual(replay(entries), 4)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            Case(Op.ADD, -1, 0)
        with self.assertRaises(ValueError):
            Case(Op.ADD, 1 << 32, 0)


class GuidanceTests(unittest.TestCase):
    def test_model_learns_labels_without_bug_feedback(self):
        model = NeuralSelector(0, 2)
        for i in range(512):
            positive = i % 2 == 0
            model.observe(np.array([1.0, 0.0]) if positive else np.array([0.0, 1.0]), positive)
        scores = model.predict(np.eye(2))
        self.assertGreater(scores[0], 0.8)
        self.assertLess(scores[1], 0.2)

    def test_features_and_functional_bins(self):
        case = Case(Op.SRA, 0x80000000, 33)
        self.assertTrue(np.isfinite(features(case)).all())
        self.assertIn('SRA:shift:1', coverage_bins(case, oracle(case)))
        self.assertNotEqual(coverage_bins(case, 0), coverage_bins(case, oracle(case)))
        self.assertEqual(Case.from_dict(case.to_dict()), case)

    def test_seeded_trials_reproduce(self):
        a = run_trial('neural', 7, 96, 8)
        b = run_trial('neural', 7, 96, 8)
        a.pop('seconds'); b.pop('seconds')
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
