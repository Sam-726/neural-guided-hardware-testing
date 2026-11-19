# Explaining the project

## The question

Hardware bugs sometimes appear only with particular bit patterns. A fuzzer generates many inputs to look for cases where the design behaves incorrectly. This experiment asks whether a neural selector can spend a fixed test budget more effectively than random selection or a simple heuristic.

## Follow one input

```text
Operation: SRA (arithmetic right shift)
Operand A: 0x80000000
Operand B: 0x00000001
Expected:  0xc0000000
Faulty:    0x40000000
```

The high bit of A represents a negative value. An arithmetic right shift must copy that sign bit. The deliberately faulty circuit fills the left side with zeroes instead. The Python oracle predicts the expected value; the simulator exposes the faulty HDL result.

To see this in code, follow `oracle()` in `hardware.py`, the `SRA` HDL branch, then `run_trial()` in `experiment.py`. The example is valid for the fault but the saved reduced counterexample may use different operands.

## What is the AI part?

The neural network sees features describing the operation and operands. After an input is executed, it receives a label: did this execution cover at least one previously unseen functional bin? It uses these examples to rank the next pool of candidates. It never sees a label saying which deliberate fault an input would expose.

The learner predicts useful candidates. It does not read processor source code, use an LLM, generate full machine-code programs, or discover real processor vulnerabilities.

## What did the experiment show?

The heuristic covered the most bins. The neural selector was close to random on average and more variable. All three found all four seeded faults across ten seeds. The honest conclusion is that this small experiment does not justify using a neural network over the simpler heuristic.

One reason to investigate is label staleness: an input that was novel early in a run may no longer be novel later, but its old positive training label remains in replay. Another limitation is the coverage definition: some bins can be predicted directly from the input, giving the count heuristic an advantage. These are hypotheses for follow-up experiments, not proven explanations.

## Things to understand before discussing it

1. Why does adding one to `0xffffffff` give zero in a 32-bit ALU?
2. How do signed SLT and unsigned SLTU differ for `0xffffffff` and zero?
3. Why are shifts masked to five bits for RV32I?
4. What information does the neural selector receive before and after execution?
5. Why do equal execution budgets and shared candidate pools matter?
6. Why are functional-bin counts not the same as code coverage?
7. Why does detecting intentional mutations not establish a new vulnerability?
8. What result would be needed before claiming the learner improves bug finding?

## A possible next step

Choose one extension, write down its expected effect before running it, and preserve the baseline results. For example, compare the current historical-label replay buffer against one that relabels examples using the current coverage state. This would be a concrete contribution to make and explain yourself, whether or not it improves the result.
