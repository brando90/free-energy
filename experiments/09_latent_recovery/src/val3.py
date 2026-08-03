import json, sys, io, contextlib, traceback
sys.path.insert(0, ".")
res_out = []
try:
    from validator import main as vmain
    for res in ["results_llama", "results_mistral"]:
        for fam in ["paraphrase", "negstep", "falsehood"]:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    vmain(res, f"perturbed_{fam}", f"_{fam}")
            except Exception as e:
                res_out.append(f"vmain {res}/{fam} ERR: {repr(e)}")
    def show(res, fam, label):
        try: d = json.load(open(f"../{res}/validated_summary_{fam}.json"))
        except Exception as e: res_out.append(f"{label}: no summary ({e})"); return
        for p in ["early","mid","late"]:
            c=d.get(p,{}); n=c.get("n",0)
            if n: res_out.append("%-20s %-6s n=%3d valid=%.3f poisoned=%.3f doubt=%.3f"%(label,p,n,c.get("valid_rederivation",0)/n,c.get("poisoned",0)/n,c.get("acknowledged",0)/n))
    res_out.append("### LLAMA-3.1-8B ###")
    for fam in ["paraphrase","negstep","falsehood"]: show("results_llama",fam,fam)
    res_out.append("### MISTRAL-7B (low n) ###")
    for fam in ["paraphrase","negstep","falsehood"]: show("results_mistral",fam,fam)
except Exception:
    res_out.append("TOP-LEVEL TRACEBACK:\n"+traceback.format_exc())
open("/tmp/valout.txt","w").write("\n".join(res_out)+"\n")
