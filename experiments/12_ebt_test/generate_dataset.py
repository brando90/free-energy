#!/usr/bin/env python3
"""Generate harder goal-directed synthetic deduction chains."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizer import SyntheticTokenizer, fact_symbols


LETTERS = fact_symbols(128)
RULE_TYPES = ("unary", "or", "and")


@dataclass(frozen=True)
class Rule:
    premises: tuple[str, ...]
    op: str
    consequent: str
    proof_premise: str | None = None
    distractor: bool = False

    def tokens(self) -> list[str]:
        if self.op == "unary":
            return [self.premises[0], "->", self.consequent]
        return [self.premises[0], self.op, self.premises[1], "->", self.consequent]

    def key(self) -> tuple[tuple[str, ...], str, str]:
        return (self.premises, self.op, self.consequent)

    def to_json(self) -> dict[str, object]:
        return {
            "premises": list(self.premises),
            "op": self.op,
            "consequent": self.consequent,
            "proof_premise": self.proof_premise,
            "distractor": self.distractor,
            "text": " ".join(self.tokens()),
        }


def unused_letter(rng: random.Random, excluded: set[str]) -> str:
    choices = [letter for letter in LETTERS if letter not in excluded]
    if not choices:
        raise ValueError("Ran out of letters while building example")
    return rng.choice(choices)


def add_trace_fact(trace: list[str], fact: str) -> None:
    if fact not in trace:
        trace.append(fact)


def fireable(rule: Rule, known: set[str]) -> bool:
    if rule.op == "unary":
        return rule.premises[0] in known
    if rule.op == "or":
        return rule.premises[0] in known or rule.premises[1] in known
    if rule.op == "and":
        return rule.premises[0] in known and rule.premises[1] in known
    raise ValueError(f"Unknown rule op: {rule.op}")


def closure(starts: Iterable[str], rules: Iterable[Rule]) -> set[str]:
    known = set(starts)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.consequent not in known and fireable(rule, known):
                known.add(rule.consequent)
                changed = True
    return known


def render_prompt(starts: list[str], rules: list[Rule], goal: str) -> list[str]:
    prompt = ["Start", "["]
    for idx, start in enumerate(starts):
        if idx:
            prompt.append(",")
        prompt.append(start)
    prompt.extend(["]", "<SEP>", "Rules", "["])
    for idx, rule in enumerate(rules):
        if idx:
            prompt.append(",")
        prompt.extend(rule.tokens())
    prompt.extend(["]", "<SEP>", "Goal", goal])
    return prompt


def render_target(trace: list[str]) -> list[str]:
    # Canonical target is fact-only. The prompt still contains implication
    # arrows in rules; the answer does not get free structural arrow tokens.
    return list(trace)


def choose_rule_type(rng: random.Random, step_idx: int, depth: int) -> str:
    if depth >= 3 and step_idx in {1, 2}:
        return rng.choice(RULE_TYPES)
    return rng.choices(RULE_TYPES, weights=[0.48, 0.27, 0.25], k=1)[0]


def build_proof(
    rng: random.Random,
    depth: int,
    starts: list[str],
) -> tuple[list[str], list[Rule], str]:
    used_letters = set(starts)
    known = set(starts)
    trace: list[str] = []
    rules: list[Rule] = []

    current = rng.choice(starts)
    add_trace_fact(trace, current)

    for step_idx in range(depth):
        consequent = unused_letter(rng, used_letters)
        used_letters.add(consequent)
        rule_type = choose_rule_type(rng, step_idx, depth)

        if rule_type == "unary":
            premise = current
            add_trace_fact(trace, premise)
            rule = Rule((premise,), "unary", consequent, proof_premise=premise)
        elif rule_type == "or":
            used_premise = current
            other = unused_letter(rng, used_letters | {used_premise})
            # Keep the unused OR branch unavailable unless it was already known.
            premises = (used_premise, other) if rng.random() < 0.5 else (other, used_premise)
            add_trace_fact(trace, used_premise)
            rule = Rule(premises, "or", consequent, proof_premise=used_premise)
        else:
            left = current
            right = rng.choice(sorted(known - {left})) if len(known) > 1 else None
            if right is None:
                right = unused_letter(rng, used_letters | {left})
                used_letters.add(right)
                starts.append(right)
                known.add(right)
            if rng.random() < 0.5:
                premises = (left, right)
            else:
                premises = (right, left)
            add_trace_fact(trace, premises[0])
            add_trace_fact(trace, premises[1])
            rule = Rule(premises, "and", consequent)

        rules.append(rule)
        known.add(consequent)
        trace.append(consequent)
        current = consequent

    return trace, rules, current


def make_distractor_rule(
    rng: random.Random,
    starts: list[str],
    proof_rules: list[Rule],
    goal: str,
    used_rule_keys: set[tuple[tuple[str, ...], str, str]],
) -> Rule:
    proof_facts = {fact for rule in proof_rules for fact in (*rule.premises, rule.consequent)}
    proof_facts.update(starts)
    allowed_rhs = [letter for letter in LETTERS if letter != goal and letter not in proof_facts]
    if not allowed_rhs:
        allowed_rhs = [letter for letter in LETTERS if letter != goal]

    for _ in range(200):
        op = rng.choice(RULE_TYPES)
        consequent = rng.choice(allowed_rhs)
        if op == "unary":
            premises = (rng.choice(LETTERS),)
        else:
            a, b = rng.sample(LETTERS, 2)
            premises = (a, b)
        rule = Rule(premises, op, consequent, distractor=True)
        if rule.key() not in used_rule_keys:
            used_rule_keys.add(rule.key())
            return rule
    raise ValueError("Could not create unique distractor rule")


def verify_example(starts: list[str], rules: list[Rule], trace: list[str], goal: str) -> None:
    if not trace or trace[-1] != goal:
        raise ValueError("Proof trace must end at the goal")
    known = set(starts)
    if trace[0] not in known:
        raise ValueError("Proof trace must begin with a starting fact")

    trace_idx = 0
    while trace_idx < len(trace):
        fact = trace[trace_idx]
        if fact in known:
            trace_idx += 1
            continue
        producers = [rule for rule in rules if rule.consequent == fact and fireable(rule, known)]
        if not producers:
            raise ValueError(f"Trace fact {fact} is not derivable from known={sorted(known)}")
        known.add(fact)
        trace_idx += 1

    if goal not in closure(starts, rules):
        raise ValueError("Goal is not derivable")


def make_chain_example(
    rng: random.Random,
    index: int,
    split: str,
    min_depth: int,
    max_depth: int,
    min_starts: int,
    max_starts: int,
    min_distractors: int,
    max_distractors: int,
    max_full_tokens: int,
) -> dict:
    depth = rng.randint(min_depth, max_depth)
    num_starts = rng.randint(min_starts, max_starts)
    starts = rng.sample(LETTERS, num_starts)
    trace, proof_rules, goal = build_proof(rng, depth, starts)

    used_rule_keys = {rule.key() for rule in proof_rules}
    distractor_count = rng.randint(min_distractors, max_distractors)
    distractors = [
        make_distractor_rule(rng, starts, proof_rules, goal, used_rule_keys)
        for _ in range(distractor_count)
    ]

    tokenizer = SyntheticTokenizer.default()
    target_tokens = render_target(trace)
    while True:
        rules = [*proof_rules, *distractors]
        rng.shuffle(rules)
        verify_example(starts, rules, trace, goal)
        prompt_tokens = render_prompt(starts, rules, goal)
        full_len = len(prompt_tokens) + 1 + len(target_tokens) + 1
        if full_len <= max_full_tokens:
            break
        if len(distractors) <= min_distractors:
            raise ValueError(
                f"Could not fit example within max_full_tokens={max_full_tokens}; "
                f"full_len={full_len}, depth={depth}, distractors={len(distractors)}"
            )
        distractors.pop()

    return {
        "id": index,
        "split": split,
        "depth": depth,
        "num_distractors": len(distractors),
        "full_length": len(prompt_tokens) + 1 + len(target_tokens) + 1,
        "starts": starts,
        "start": starts[0],
        "goal": goal,
        "query": goal,
        "rules": [rule.to_json() for rule in rules],
        "proof_rules": [rule.to_json() for rule in proof_rules],
        "distractor_rules": [rule.to_json() for rule in distractors],
        "rule_types": [rule.op for rule in proof_rules],
        "facts_rules": tokenizer.decode(tokenizer.encode(prompt_tokens)),
        "prompt_tokens": prompt_tokens,
        "path_tokens": trace,
        "target_tokens": target_tokens,
    }


def generate_dataset(
    train_samples: int,
    val_samples: int,
    test_samples: int,
    seed: int,
    output_path: Path,
    min_depth: int,
    max_depth: int,
    min_starts: int,
    max_starts: int,
    min_distractors: int,
    max_distractors: int,
    max_full_tokens: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits = (
        (train_samples, "train"),
        (val_samples, "val"),
        (test_samples, "test"),
    )

    current = 0
    split_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    rule_counts: Counter[str] = Counter()
    max_prompt_len = 0
    max_target_len = 0
    max_full_len = 0
    distractor_counts: Counter[int] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for split_count, split_name in splits:
            for _ in range(split_count):
                row = make_chain_example(
                    rng=rng,
                    index=current,
                    split=split_name,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    min_starts=min_starts,
                    max_starts=max_starts,
                    min_distractors=min_distractors,
                    max_distractors=max_distractors,
                    max_full_tokens=max_full_tokens,
                )
                current += 1
                split_counts[split_name] += 1
                depth_counts[int(row["depth"])] += 1
                rule_counts.update(row["rule_types"])
                distractor_counts[int(row["num_distractors"])] += 1
                max_prompt_len = max(max_prompt_len, len(row["prompt_tokens"]))
                max_target_len = max(max_target_len, len(row["target_tokens"]))
                max_full_len = max(max_full_len, int(row["full_length"]))
                handle.write(json.dumps(row) + "\n")

    return {
        "splits": dict(split_counts),
        "depth_histogram": dict(sorted(depth_counts.items())),
        "proof_rule_type_histogram": dict(rule_counts),
        "distractor_histogram": dict(sorted(distractor_counts.items())),
        "max_prompt_len": max_prompt_len,
        "max_target_len": max_target_len,
        "max_full_len": max_full_len,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None, help="Compatibility alias for train samples")
    parser.add_argument("--num-train", type=int, default=300_000)
    parser.add_argument("--num-val", type=int, default=1_000)
    parser.add_argument("--num-test", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "EBT-main"
        / "data"
        / "nlp"
        / "synthetic_chain_dataset"
        / "synthetic_chain_dataset.jsonl",
    )
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--min-starts", type=int, default=1)
    parser.add_argument("--max-starts", type=int, default=5)
    parser.add_argument("--min-distractors", type=int, default=35)
    parser.add_argument("--max-distractors", type=int, default=90)
    parser.add_argument("--max-full-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_path.name.endswith(".jsonl"):
        raise ValueError("output_path must be a .jsonl file")
    if args.num_samples is not None:
        args.num_train = args.num_samples
    if args.min_depth < 1 or args.max_depth < args.min_depth:
        raise ValueError("Depth bounds must satisfy 1 <= min_depth <= max_depth")
    if args.min_starts < 1 or args.max_starts < args.min_starts:
        raise ValueError("Start bounds must satisfy 1 <= min_starts <= max_starts")
    if args.min_distractors < 0 or args.max_distractors < args.min_distractors:
        raise ValueError("Distractor bounds must satisfy 0 <= min_distractors <= max_distractors")
    if args.max_full_tokens < 1:
        raise ValueError("max_full_tokens must be positive")

    summary = generate_dataset(
        train_samples=args.num_train,
        val_samples=args.num_val,
        test_samples=args.num_test,
        seed=args.seed,
        output_path=args.output_path,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_starts=args.min_starts,
        max_starts=args.max_starts,
        min_distractors=args.min_distractors,
        max_distractors=args.max_distractors,
        max_full_tokens=args.max_full_tokens,
    )
    total = sum(summary["splits"].values())
    print(f"wrote={args.output_path} total={total}")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
