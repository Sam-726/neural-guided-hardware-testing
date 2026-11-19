import argparse
import csv
import html
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from amaranth.sim import Simulator

from .guidance import Selector, candidates, coverage_bins
from .hardware import Case, MUTANTS, TestRig, drive, oracle, read

STRATEGIES = ("random", "coverage", "neural")


def run_trial(strategy, seed, budget, pool_size):
    rig = TestRig()
    sim = Simulator(rig)
    sampler = np.random.default_rng(seed)
    selector = Selector(strategy, seed)
    seen, first_hits, examples, curve = set(), {}, {}, []

    async def bench(ctx):
        for step in range(1, budget + 1):
            case = selector.choose(candidates(sampler, pool_size))
            drive(ctx, rig, case)
            await ctx.delay(1e-9)
            results = read(ctx, rig)
            expected = oracle(case)
            if results["correct"] != expected:
                raise AssertionError(f"Correct RTL disagrees with oracle: {case}")
            bins = coverage_bins(case, results["correct"])
            novel = bool(bins - seen)
            seen.update(bins)
            selector.observe(case, novel)
            for mutant in MUTANTS:
                if results[mutant] != expected and mutant not in first_hits:
                    first_hits[mutant] = step
                    examples[mutant] = {**case.to_dict(), "expected": f"0x{expected:08x}",
                                        "actual": f"0x{results[mutant]:08x}"}
            if step == 1 or step % 32 == 0 or step == budget:
                curve.append({"tests": step, "bins": len(seen)})

    sim.add_testbench(bench)
    started = time.perf_counter()
    sim.run()
    return {"strategy": strategy, "seed": seed, "budget": budget, "coverage_bins": len(seen),
            "seconds": time.perf_counter() - started, "first_hits": first_hits,
            "examples": examples, "curve": curve}


def reduce_case(mutant, original):
    """Greedily clear operand bits while preserving the observed mismatch."""
    rig, best = TestRig(), original
    sim = Simulator(rig)

    async def bench(ctx):
        nonlocal best
        drive(ctx, rig, best)
        await ctx.delay(1e-9)
        if read(ctx, rig)[mutant] == oracle(best):
            raise ValueError("Input does not reproduce the named mutation")
        for operand in ("a", "b"):
            for bit in range(31, -1, -1):
                value = getattr(best, operand)
                if not value & (1 << bit):
                    continue
                trial = Case(best.op, value & ~(1 << bit), best.b) if operand == "a" else Case(best.op, best.a, value & ~(1 << bit))
                drive(ctx, rig, trial)
                await ctx.delay(1e-9)
                results = read(ctx, rig)
                assert results["correct"] == oracle(trial)
                if results[mutant] != oracle(trial):
                    best = trial

    sim.add_testbench(bench)
    sim.run()
    return best


def replay(examples, waveform=None):
    rig, failures = TestRig(), []
    sim = Simulator(rig)

    async def bench(ctx):
        for entry in examples:
            case, mutant = Case.from_dict(entry), entry["mutant"]
            drive(ctx, rig, case)
            await ctx.delay(1e-9)
            results = read(ctx, rig)
            expected = oracle(case)
            if results["correct"] != expected or results[mutant] == expected:
                failures.append(entry)

    sim.add_testbench(bench)
    if waveform:
        with sim.write_vcd(str(waveform)):
            sim.run()
    else:
        sim.run()
    if failures:
        raise AssertionError(f"{len(failures)} counterexamples failed replay")
    return len(examples)


def write_report(payload, out):
    rows = []
    for strategy in STRATEGIES:
        trials = [r for r in payload["trials"] if r["strategy"] == strategy]
        row = {"strategy": strategy, "runs": len(trials),
               "mean_bins": round(statistics.mean(r["coverage_bins"] for r in trials), 2),
               "stdev_bins": round(statistics.stdev(r["coverage_bins"] for r in trials), 2) if len(trials) > 1 else 0,
               "mean_mutants_found": round(statistics.mean(len(r["first_hits"]) for r in trials), 2),
               "mean_seconds": round(statistics.mean(r["seconds"] for r in trials), 3)}
        rows.append(row)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    content = """<!doctype html><meta charset="utf-8"><title>Neural-guided ALU testing</title>
    <style>body{font:17px/1.6 system-ui;max-width:1000px;margin:60px auto;padding:0 24px;color:#182230}
    h1{font-size:40px;line-height:1.2}h2{margin-top:38px}table{border-collapse:collapse;width:100%}
    th,td{text-align:left;border-bottom:1px solid #d5dce4;padding:12px}th{background:#f1f5f9}
    .note{background:#edf3fa;padding:20px;border-left:4px solid #366ba0}code{font-size:14px}</style>
    <p>EXPERIMENT REPORT</p><h1>Can a neural selector improve hardware test coverage?</h1>
    <p>A small RV32I-style arithmetic unit, four deliberate HDL mutations, three test strategies.</p>"""
    content += f'<p>{len(payload["config"]["seeds"])} seeds · {payload["config"]["budget"]} executed tests per run · {payload["config"]["pool_size"]} candidates per decision.</p>'
    content += '<table><tr><th>Strategy</th><th>Functional bins<br>mean ± sample SD</th><th>Mutants found<br>mean / 4</th><th>Time / run</th></tr>'
    for r in rows:
        content += f'<tr><td>{r["strategy"]}</td><td>{r["mean_bins"]} ± {r["stdev_bins"]}</td><td>{r["mean_mutants_found"]}</td><td>{r["mean_seconds"]} s</td></tr>'
    content += '</table><h2>How to interpret this</h2><div class="note">Higher functional-bin counts indicate more diversity under this hand-written metric. They are not HDL branch coverage or a percentage of all reachable states. The four bugs are intentionally seeded; no real processor vulnerability was discovered. A neural method is not assumed to beat the heuristic.</div>'
    content += '<h2>First detection by mutation</h2><table><tr><th>Strategy</th><th>Mutation</th><th>Detected runs</th><th>Median tests<br>among detected runs</th></tr>'
    for strategy in STRATEGIES:
        trials = [r for r in payload['trials'] if r['strategy'] == strategy]
        for mutant in MUTANTS:
            hits = [r['first_hits'][mutant] for r in trials if mutant in r['first_hits']]
            median = statistics.median(hits) if hits else 'not detected'
            content += f'<tr><td>{strategy}</td><td>{mutant}</td><td>{len(hits)} / {len(trials)}</td><td>{median}</td></tr>'
    content += '</table><p>Undetected runs are censored at the test budget and excluded from detection medians. Always read medians with the detected-run counts.</p>'
    content += '<h2>Reproducibility</h2><p>All strategies share the same seeded candidate pools and 32-test warm-up. Guided selectors have 15% random exploration. Only executed tests contribute to coverage. The neural model trains online on novelty, never mutation mismatches. Simulator and model overhead are included in run times; initialization is excluded.</p>'
    content += '<p>Raw runs and coverage curves: <a href="results.json">results.json</a> · <a href="summary.csv">summary.csv</a> · <a href="counterexamples.json">reduced counterexamples</a> · <a href="counterexamples.vcd">HDL waveforms</a></p>'
    content += f'<pre>{html.escape(json.dumps(payload["environment"], indent=2))}</pre>'
    (out / 'report.html').write_text(content)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--pool-size', type=int, default=32)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.replay:
        print(f'Replayed {replay(json.loads(args.replay.read_text()))} counterexamples successfully')
        return
    if min(args.seeds, args.budget, args.pool_size) < 1:
        parser.error('seeds, budget and pool-size must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    payload = {'config': {'seeds': list(range(args.seeds)), 'budget': args.budget, 'pool_size': args.pool_size},
               'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                               'amaranth': version('amaranth'), 'numpy': np.__version__}, 'trials': []}
    for seed in range(args.seeds):
        for strategy in STRATEGIES:
            result = run_trial(strategy, seed, args.budget, args.pool_size)
            payload['trials'].append(result)
            print(f'{strategy:8} seed={seed:2} bins={result["coverage_bins"]:3} mutants={len(result["first_hits"])}', flush=True)
    (out / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    reduced = []
    for mutant in MUTANTS:
        source = next((r['examples'][mutant] for r in payload['trials'] if mutant in r['examples']), None)
        if source:
            small = reduce_case(mutant, Case.from_dict(source))
            reduced.append({'mutant': mutant, **small.to_dict(), 'expected': f'0x{oracle(small):08x}', 'original': source})
    (out / 'counterexamples.json').write_text(json.dumps(reduced, indent=2) + '\n')
    replay(reduced, out / 'counterexamples.vcd')
    print(json.dumps(write_report(payload, out), indent=2))
    print(f'Report: {out / "report.html"}')


if __name__ == '__main__':
    main()
