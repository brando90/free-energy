#!/usr/bin/env python3
import csv, json
csv.field_size_limit(10**9)
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/grading"
rows = list(csv.DictReader(open(OUT + "/cells_final.csv")))

print("===== GPT-4 sbs grid (pipeline emulation = published-table method) =====")
print(f"{'ds':6s} {'pos':8s} {'pert':7s} {'ty':2s} | {'pipe_n':>6s} {'pipe%':>6s} {'unann':>5s} | {'strict_n':>8s} {'strict%':>7s} | {'cc_n':>5s} {'cc%':>6s}")
for r in rows:
    if r['model'] == 'gpt-4-0314' and r['style'] == 'sbs':
        print(f"{r['dataset']:6s} {r['position']:8s} {r['perturbation']:7s} {r['typo10']:2s} | "
              f"{r['pipe_n']:>6s} {r['pipe_rate']:>6s} {r['pipe_unannot']:>5s} | "
              f"{r['strict_n']:>8s} {r['strict_rate']:>7s} | {r['cc_n']:>5s} {r['cc_rate']:>6s}")

print("\n===== paper Exp3 comparison (GPT-4, perturbation=random, sbs, pipeline) =====")
paper = {('awps','calc'):(77.73,247), ('asdiv','calc'):(74.34,226), ('svamp','calc'):(76.54,243), ('gsm8k','calc'):(78.57,224),
         ('awps','copy'):(99.32,None), ('asdiv','copy'):(97.45,None), ('svamp','copy'):(97.71,None), ('gsm8k','copy'):(94.98,None),
         ('awps','propcalc'):(32.54,None), ('asdiv','propcalc'):(52.71,None), ('svamp','propcalc'):(54.47,None), ('gsm8k','propcalc'):(26.45,None)}
for r in rows:
    if r['model']=='gpt-4-0314' and r['style']=='sbs' and r['perturbation']=='random' and r['typo10']=='0':
        k = (r['dataset'], r['position'])
        p = paper.get(k)
        if p:
            d = round(float(r['pipe_rate']) - p[0], 2)
            print(f"  {k[0]:6s} {k[1]:8s} paper={p[0]:6.2f}/n={p[1]} | release pipe={r['pipe_rate']:>6s}/n={r['pipe_n']:>3s}  diff={d:+.2f}")

print("\n===== other models, sbs, perturbation=random (cross-model table) =====")
for mdl in ['gpt-3.5-turbo-0301','claude-3-opus-20240229','meta-llama/Llama-3-70b-chat-hf']:
    print(f"-- {mdl}")
    for r in rows:
        if r['model']==mdl and r['style']=='sbs' and r['perturbation']=='random':
            print(f"  {r['dataset']:6s} {r['position']:8s} typo={r['typo10']} pipe={r['pipe_rate']:>6s}/n={r['pipe_n']:>3s} unannot={r['pipe_unannot']:>3s} strict={r['strict_rate']:>6s}/n={r['strict_n']:>3s}")

print("\n===== GPT-4 sbs add1 vs add101 (Exp1) =====")
for r in rows:
    if r['model']=='gpt-4-0314' and r['style']=='sbs' and r['perturbation'] in ('add1','add101'):
        print(f"  {r['dataset']:6s} {r['position']:8s} {r['perturbation']:7s} pipe={r['pipe_rate']:>6s}/n={r['pipe_n']:>3s} unannot={r['pipe_unannot']:>3s} strict={r['strict_rate']:>6s}/n={r['strict_n']:>3s} cc={r['cc_rate']:>6s}")

print("\n===== sbs_recovery (unpublished-in-repo recovery-prompt cells, gpt-4) =====")
for r in rows:
    if r['model']=='gpt-4-0314' and r['style']=='sbs_recovery':
        print(f"  {r['dataset']:6s} {r['position']:8s} {r['perturbation']:7s} typo={r['typo10']} pipe={r['pipe_rate']:>6s}/n={r['pipe_n']:>3s}")

ct = json.load(open(OUT + "/ack_crosstab2.json"))
print("\n===== explicit_without_ack (sbs) first 12 =====")
for e in ct['examples'].get('explicit_without_ack_sbs', [])[:12]:
    print(json.dumps(e)[:420])
print("\n===== unfaithful_with_ack (sbs) all =====")
for e in ct['examples'].get('unfaithful_with_ack_sbs', []):
    print(json.dumps(e)[:420])
