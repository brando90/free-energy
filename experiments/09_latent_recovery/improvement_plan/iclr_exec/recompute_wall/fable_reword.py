import sys, json
EXP="/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
for p in (EXP+"/improvement_plan/exph", EXP+"/improvement_plan/expg"):
    sys.path.insert(0,p)
import trace_format as tf, anthropic
progs=[json.loads(l) for l in open(EXP+"/improvement_plan/iclr_exec/recompute_wall/e10_topup_fable/programs_batch2.jsonl")]
c=anthropic.Anthropic(max_retries=1)
def listing(i): return tf.make_listing_text(progs[i]["stmt_texts"])
def run(name, build, **kw):
    ref=ok=0;cat=None;last=""
    for i in range(3):
        try:
            r=c.messages.create(model="claude-fable-5",max_tokens=1200,messages=[{"role":"user","content":build(i)}],**kw)
            if r.stop_reason=="refusal": ref+=1;cat=getattr(getattr(r,"stop_details",None),"category",None)
            else: ok+=1; last="".join(b.text for b in r.content if getattr(b,"type",None)=="text")
        except Exception as e: print(f"  [{name}] ERR {str(e)[:110]}");return
    print(f"[{name}] ok={ok} refused={ref} cat={cat}"+(f"\n   sample={last[:160]!r}" if ok else ""))
# reworded: pure arithmetic worksheet, no 'program/trace/execution/print'
def b_math(i):
    return ("Evaluate the following list of arithmetic definitions in order. For each, give its value.\n\n"
            + "\n".join(progs[i]["stmt_texts"]).replace("print(","show(") +
            "\n\nList each variable and its computed value, one per line.")
run("reword_math_low", b_math, output_config={"effort":"low"})
run("reword_math_high", b_math, output_config={"effort":"high"})
