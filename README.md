# Neural-Guided Hardware Testing

Can a small neural network select more useful hardware tests than random sampling or a simple coverage heuristic?

This project builds a 32-bit arithmetic logic unit in **Amaranth HDL**, simulates it, and compares three ways to choose tests. An independent Python reference checks the hardware outputs. Four deliberately incorrect HDL variants provide reproducible bug-detection experiments.

This is a working research prototype with measured results, not a vulnerability discovery or a complete RISC-V processor.

## Run it

Python 3.12 is the tested interpreter. Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m hwfuzz --seeds 10 --budget 512 --pool-size 32
python -m hwfuzz --replay results/counterexamples.json
```

Open `results/report.html` in a browser. No account, API key, GPU, FPGA board or external simulator is required. Amaranth supplies the HDL simulator. The local environment supplied with this workspace is already installed; activate it and start with the test command.

## What is being tested?

The HDL implements ten RV32I-style integer operations: ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT and SLTU. Arithmetic wraps to 32 bits; shifts use the low five bits of the second operand. Signed comparison and arithmetic right shift use two's-complement interpretation.

It does not implement instruction decoding, registers, instruction sequences, memory, pipelines, timing analysis or a complete CPU. The operation selector is an internal testbench enum, not an ISA instruction encoding.

Four separately simulated mutations introduce realistic *classes* of implementation mistakes:

| Mutation | Incorrect behaviour |
| --- | --- |
| `sra_logical` | Arithmetic right shift uses zero fill instead of sign extension |
| `slt_unsigned` | Signed less-than compares operands as unsigned integers |
| `shift_six_bits` | Shifts use six bits of the shift amount instead of five |
| `add_saturates` | Unsigned addition saturates on carry instead of wrapping |

The correct HDL and all four mutants receive each selected input. The correct HDL must agree with the independent integer oracle on every execution; disagreement aborts the experiment. Mutation mismatches are recorded for evaluation and never supplied to the neural selector.

## How test selection works

1. Generate a pool of 32 candidate operations and operand pairs. Each operand has a 35% chance of being sampled from a fixed boundary-value list; otherwise it is uniformly sampled from all 32-bit values.
2. Select exactly one candidate to execute.
3. Simulate the HDL and measure functional-bin novelty.
4. Update the selector and repeat until the execution budget is used.

The three strategies share the same seeded candidate pools and first 32 tests:

| Strategy | Decision rule |
| --- | --- |
| `random` | Execute the first randomly generated candidate |
| `coverage` | Prefer candidates whose input-operation, operand-class and shift-amount bins have been executed least often |
| `neural` | A NumPy MLP ranks candidates by predicted probability of adding a new functional bin |

Both guided strategies have 15% random exploration after warm-up. The neural model has one 24-unit tanh hidden layer, sigmoid output and binary cross-entropy updates. It trains online on previous executed inputs and their novelty labels, using uniform replay. It is **candidate selection**, not a language model or an autoregressive instruction generator. The neural network is implemented in NumPy so the experiment remains small and inspectable.

Coverage includes operation-specific operand classes, output classes, input/output sign combinations and shift amounts. This is a hand-written **functional coverage metric**, not simulator branch coverage or a percentage of all reachable hardware states. It favours the input-count heuristic by construction; that is part of interpreting the result.

## Recorded experiment

Ten seeds (`0` through `9`), 512 executed inputs per strategy per seed, 32 candidates per decision. This is 15,360 selected inputs, each evaluated against the correct HDL and four mutant variants.

| Strategy | Mean functional bins | Sample standard deviation | Mean mutants detected / 4 |
| --- | ---: | ---: | ---: |
| Random | 295.9 | 7.20 | 4.0 |
| Coverage heuristic | 365.8 | 2.25 | 4.0 |
| Neural selector | 300.2 | 14.61 | 4.0 |

The simple coverage heuristic performed best under this metric. The neural method's small mean difference from random does **not** demonstrate a reliable advantage. All methods detected all four deliberately inserted faults in every run; these mutations are too easy to establish a bug-finding advantage for the learner.

Useful follow-up experiments include harder sequential targets, simulator-internal coverage feedback, a replay buffer that refreshes stale novelty labels, and additional targets/seeds. These are future work, not completed capabilities. No parameters or seeds were removed to improve the displayed outcome.

## Evidence and validation

- `results/results.json`: every run, first detection counts, original examples, coverage curves and environment versions.
- `results/summary.csv`: aggregate measurements.
- `results/counterexamples.json`: greedily bit-cleared examples that still reproduce each fault.
- `results/counterexamples.vcd`: waveform evidence from replay against the actual HDL simulation.
- `results/report.html`: readable experiment report with detection rates and timing.
- `tests/test_fuzzing.py`: known arithmetic vectors, HDL/reference agreement over 1,690 boundary combinations plus 1,000 seeded random inputs, mutation replay/reduction, input validation, model learning and seeded reproducibility.

Reduction clears operand bits while preserving a mismatch; it does not prove globally minimal counterexamples. First-detection medians in the report exclude undetected runs and are always shown beside detected-run counts. Timing is machine-dependent and includes candidate generation, selector work and simulation, but excludes HDL simulator initialization.

## File guide

```text
hwfuzz/hardware.py     HDL, mutations, integer oracle and test-case format
hwfuzz/guidance.py     Sampling, functional coverage and online neural selector
hwfuzz/experiment.py   Benchmarks, reduction, replay and report generation
tests/                Behavioural and reproducibility checks
results/              Recorded experiment and replay artifacts
WALKTHROUGH.md        Short explanation for discussing the project
```

## Sources

- [RISC-V RV32I specification](https://docs.riscv.org/reference/isa/v20240411/unpriv/rv32.html): arithmetic, comparison and shift semantics.
- [Amaranth 0.5.8 simulator](https://amaranth-lang.org/docs/amaranth/v0.5.8/simulator.html): HDL simulation and VCD waveforms.
- [Amaranth language guide](https://amaranth-lang.org/docs/amaranth/v0.5.8/guide.html): signal widths and signedness.
- [AI-assisted hardware fuzzing research notice](https://www.informatik.tu-darmstadt.de/media/systemsecurity/dateien_5/jobs_3/AIBasedHardwareFuzzing.pdf): motivation. This independent project is not affiliated with the lab and has not been reviewed by its researchers.

The project was created with coding-assistant support. Review the implementation and reproduce the results before describing personal contributions in an application or interview. It has not been added to an application.
