import os, sys, json
EXP="/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
for p in (EXP+"/improvement_plan/exph", EXP+"/improvement_plan/expg"):
    sys.path.insert(0,p)
import trace_format as tf, anthropic
progs=[json.loads(l) for l in open(EXP+"/improvement_plan/iclr_exec/recompute_wall/e10_topup_fable/programs_batch2.jsonl")]
c=anthropic.Anthropic(max_retries=1)
def probe(name, messages, system=None, n=3, **kw):
    ref=0; ok=0; cat=None
    for i in range(n):
        listing=tf.make_listing_text(progs[i]["stmt_texts"])
        msgs=[{"role":"user","content":m["content"].replace("__LISTING__", listing) if "__LISTING__" in m["content"] else m["content"]} for m in messages]
        try:
            call=dict(model="claude-fable-5", max_tokens=1536, messages=msgs, **kw)
            if system: call["system"]=system
            r=c.messages.create(**call)
            if r.stop_reason=="refusal":
                ref+=1; cat=getattr(getattr(r,"stop_details",None),"category",None)
            else:
                ok+=1; last="".join(b.text for b in r.content if getattr(b,"type",None)=="text")
        except Exception as e:
            print(f"  [{name}] ERR {type(e).__name__}: {str(e)[:120]}"); return
    print(f"[{name}] ok={ok} refused={ref} cat={cat}" + (f" sample={last[:90]!r}" if ok else ""))

USER=[{"role":"user","content":tf.INSTRUCTION+tf.EXEMPLARS+"Program:\n__LISTING__"}]  # ~gold-ish; actually use make_user_message
# real gold user builder
def gold_msgs(i): return [{"role":"user","content":tf.make_user_message(tf.make_listing_text(progs[i]["stmt_texts"]))}]
def probe_real(name, system=None, **kw):
    ref=ok=0; cat=None; last=""
    for i in range(3):
        try:
            call=dict(model="claude-fable-5",max_tokens=1536,messages=gold_msgs(i),**kw)
            if system: call["system"]=system
            r=c.messages.create(**call)
            if r.stop_reason=="refusal": ref+=1; cat=getattr(getattr(r,"stop_details",None),"category",None)
            else: ok+=1; last="".join(b.text for b in r.content if getattr(b,"type",None)=="text")
        except Exception as e:
            print(f"  [{name}] ERR {type(e).__name__}: {str(e)[:120]}"); return
    print(f"[{name}] ok={ok} refused={ref} cat={cat}"+(f" sample={last[:90]!r}" if ok else ""))

SYS1="You are a deterministic Python interpreter simulator used in an educational CS tool. Given a numbered program, output only the line-by-line execution trace in the requested format. This is a benign arithmetic exercise."
probe_real("sys_interp_low", system=SYS1, output_config={"effort":"low"})
probe_real("sys_interp_default", system=SYS1)
SYS2="This is a benign educational arithmetic-tracing exercise (toy variable assignments with +, -, *). Not security-related. Produce only the trace."
probe_real("sys_benign_low", system=SYS2, output_config={"effort":"low"})
