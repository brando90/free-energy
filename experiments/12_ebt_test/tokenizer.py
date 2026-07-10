#!/usr/bin/env python3
"""Small local tokenizer for the synthetic deduction-chain task."""

from __future__ import annotations

from dataclasses import dataclass


def fact_symbols(count: int = 128) -> list[str]:
    symbols: list[str] = []
    idx = 0
    while len(symbols) < count:
        n = idx
        chars: list[str] = []
        while True:
            chars.append(chr(ord("A") + (n % 26)))
            n = n // 26 - 1
            if n < 0:
                break
        symbols.append("".join(reversed(chars)))
        idx += 1
    return symbols


@dataclass(frozen=True)
class SyntheticTokenizer:
    tokens: tuple[str, ...]

    @classmethod
    def default(cls) -> "SyntheticTokenizer":
        token_list: list[str] = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            *fact_symbols(128),
            "->",
            "and",
            "or",
            "Start",
            "Rules",
            "Goal",
            "Prove",
            "<SEP>",
            "[",
            "]",
            ",",
            "|",
            "fact:",
            "rule:",
            "query:",
        ]
        return cls(tokens=tuple(token_list))

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.tokens)})

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(self, tokens: str | list[str]) -> list[int]:
        if isinstance(tokens, str):
            token_list = tokens.split()
        else:
            token_list = tokens
        return [self.token_to_id.get(token, self.unk_token_id) for token in token_list]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return " ".join(self.tokens[int(i)] for i in token_ids)

    def decode_to_tokens(self, token_ids: list[int] | tuple[int, ...]) -> list[str]:
        return [self.tokens[int(i)] for i in token_ids]
