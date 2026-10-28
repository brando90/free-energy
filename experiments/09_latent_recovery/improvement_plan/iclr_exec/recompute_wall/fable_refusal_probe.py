import os, sys, json
EXP="/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
for p in (EXP+"/improvement_plan/exph", EXP+"/improvement_plan/expg"):
    sys.path.insert(0,p)
import exph_common as C, trace_format as tf, anthropic
prog=[json.loads(l) for l in open(EXP+"/improvement_plan/iclr_exec/recompute_wall/e10_topup_fable/programs_batch2.jsonl")][0]
listing=tf.make_listing_text(prog["stmt_texts"])
gold_user=tf.make_user_message(listing)
print("=== GOLD USER MESSAGE (first 600 chars) ===")
print(gold_user[:600])
c=anthropic.Anthropic(max_retries=1)
def probe(name, messages, **kw):
    try:
        r=c.messages.create(model="claude-fable-5", max_tokens=kw.pop("max_tokens",1536), messages=messages, **kw)
        txt="".join(b.text for b in r.content if getattr(b,"type",None)=="text")
        cat=getattr(getattr(r,"stop_details",None),"category",None)
        print(f"[{name}] stop={r.stop_reason} cat={cat} out={r.usage.output_tokens} txt={txt[:100]!r}")
    except Exception as e:
        print(f"[{name}] ERR {type(e).__name__}: {str(e)[:150]}")
# a) exact gold user, effort low
probe("gold_effort_low", [{"role":"user","content":gold_user}], output_config={"effort":"low"})
# b) gold user, NO output_config (default effort)
probe("gold_default", [{"role":"user","content":gold_user}])
# c) gold user, effort high
probe("gold_effort_high", [{"role":"user","content":gold_user}], output_config={"effort":"high"})
# d) just the instruction preamble without exemplars? print instruction head
print("=== tf.INSTRUCTION head ===")
print(tf.INSTRUCTION[:400])
