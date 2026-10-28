#!/usr/bin/env python3
"""CPU self-test: imports, pool partition, prefix/probe/render plumbing."""
import e16_lib as L

for mk in ("qwen7b", "llama8b"):
    pool = L.load_pool(mk)
    parts = {f: {k: len(v) for k, v in p.items()} for f, p in pool.part.items()}
    print(mk, "programs:", len(pool.programs),
          "| fams:", {f: len(v) for f, v in pool.by_fam.items()},
          "| parts:", parts)
    for f, p in pool.part.items():
        assert not (set(p["extract"]) & set(p["eval"])), ("overlap", mk, f)
    m = next(iter(pool.by_fam["depth_k1_bare"].values()))
    up = L.unperturbed_prefix(m)
    assert up.rstrip("\n").split("\n")[-1] == m["original_trace_line"], "unpert prefix"
    prog = pool.programs[m["program_id"]]
    print("  probe q:", L.probe_question(prog, m),
          "| planted:", m["planted_value"], "true:", m["true_value"])
    assert m["prefix_text"].rstrip("\n").split("\n")[-1] == m["injected_trace_line"], \
        "prefix last line != injected line"

from transformers import AutoTokenizer
name, rev, _ = L.MODELS["qwen7b"]
tok = AutoTokenizer.from_pretrained(name, revision=rev)
pool = L.load_pool("qwen7b")
m = next(iter(pool.by_fam["depth_k1_bare"].values()))
prog = pool.programs[m["program_id"]]
um = L.user_msg_for(prog)
ids = L.render_continue(tok, um, m["prefix_text"])
a, b = L.planted_line_span(tok, um, m["prefix_text"])
probe = L.render_probe(tok, um, m["prefix_text"], L.probe_question(prog, m))
assert 0 < a < b <= len(ids), ("span", a, b, len(ids))
tail = tok.decode(ids[a:b])
assert m["planted_var"] in tail and str(m["planted_value"]) in tail, ("span text", tail)
print("render ok: cont ids", len(ids), "| span", (a, b), "->", repr(tail[:60]),
      "| probe ids", len(probe))
print("e11 seams:", sorted(L.e11.paths("/tmp/x").keys()),
      "| cont_run_id:", L.e11.cont_run_id("p", "f", 0, 1)[:12])
print("SELFTEST OK")
