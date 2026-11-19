"""Synthesizable Amaranth HDL target and an independent integer oracle.

This is a combinational ALU, not a processor or instruction decoder.
Operation numbers below are local testbench selectors, not RISC-V encodings.
"""
from dataclasses import dataclass
from enum import IntEnum

from amaranth import Elaboratable, Module, Signal

MASK = (1 << 32) - 1


class Op(IntEnum):
    ADD = 0
    SUB = 1
    AND = 2
    OR = 3
    XOR = 4
    SLL = 5
    SRL = 6
    SRA = 7
    SLT = 8
    SLTU = 9


@dataclass(frozen=True)
class Case:
    op: Op
    a: int
    b: int

    def __post_init__(self):
        if not isinstance(self.op, Op) or not (0 <= self.a <= MASK and 0 <= self.b <= MASK):
            raise ValueError("Use a valid Op and two unsigned 32-bit operands")

    def to_dict(self):
        return {"op": self.op.name, "a": f"0x{self.a:08x}", "b": f"0x{self.b:08x}"}

    @classmethod
    def from_dict(cls, value):
        return cls(Op[value["op"]], int(value["a"], 16), int(value["b"], 16))


def signed32(value):
    return value - (1 << 32) if value & (1 << 31) else value


def oracle(case):
    a, b, op = case.a, case.b, case.op
    if op == Op.ADD:
        value = a + b
    elif op == Op.SUB:
        value = a - b
    elif op == Op.AND:
        value = a & b
    elif op == Op.OR:
        value = a | b
    elif op == Op.XOR:
        value = a ^ b
    elif op == Op.SLL:
        value = a << (b & 31)
    elif op == Op.SRL:
        value = a >> (b & 31)
    elif op == Op.SRA:
        value = signed32(a) >> (b & 31)
    elif op == Op.SLT:
        value = int(signed32(a) < signed32(b))
    else:
        value = int(a < b)
    return value & MASK


MUTANTS = ("sra_logical", "slt_unsigned", "shift_six_bits", "add_saturates")


class ALU(Elaboratable):
    """RV32I integer operation semantics with optional seeded implementation fault."""

    def __init__(self, mutation=None):
        if mutation is not None and mutation not in MUTANTS:
            raise ValueError(f"Unknown mutation: {mutation}")
        self.mutation = mutation
        self.op = Signal(range(len(Op)))
        self.a, self.b, self.y = Signal(32), Signal(32), Signal(32)

    def elaborate(self, platform):
        m = Module()
        a, b, y = self.a, self.b, self.y
        shift = b[:6] if self.mutation == "shift_six_bits" else b[:5]
        m.d.comb += y.eq(0)
        with m.Switch(self.op):
            with m.Case(Op.ADD):
                if self.mutation == "add_saturates":
                    total = Signal(33)
                    m.d.comb += total.eq(a + b)
                    with m.If(total[32]):
                        m.d.comb += y.eq(MASK)
                    with m.Else():
                        m.d.comb += y.eq(total)
                else:
                    m.d.comb += y.eq(a + b)
            with m.Case(Op.SUB):
                m.d.comb += y.eq(a - b)
            with m.Case(Op.AND):
                m.d.comb += y.eq(a & b)
            with m.Case(Op.OR):
                m.d.comb += y.eq(a | b)
            with m.Case(Op.XOR):
                m.d.comb += y.eq(a ^ b)
            with m.Case(Op.SLL):
                m.d.comb += y.eq(a << shift)
            with m.Case(Op.SRL):
                m.d.comb += y.eq(a >> shift)
            with m.Case(Op.SRA):
                source = a if self.mutation == "sra_logical" else a.as_signed()
                m.d.comb += y.eq(source >> shift)
            with m.Case(Op.SLT):
                if self.mutation == "slt_unsigned":
                    m.d.comb += y.eq(a < b)
                else:
                    m.d.comb += y.eq(a.as_signed() < b.as_signed())
            with m.Case(Op.SLTU):
                m.d.comb += y.eq(a < b)
        return m


class TestRig(Elaboratable):
    def __init__(self):
        self.targets = {"correct": ALU(), **{name: ALU(name) for name in MUTANTS}}

    def elaborate(self, platform):
        m = Module()
        for name, target in self.targets.items():
            m.submodules[name] = target
        return m


def drive(ctx, rig, case):
    for target in rig.targets.values():
        ctx.set(target.op, int(case.op))
        ctx.set(target.a, case.a)
        ctx.set(target.b, case.b)


def read(ctx, rig):
    return {name: ctx.get(target.y) for name, target in rig.targets.items()}
