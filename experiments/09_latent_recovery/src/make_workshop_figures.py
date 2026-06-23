"""Generate workshop paper figures using standalone PGF/TikZ.

Matplotlib is not available in the lightweight paper-editing environment, so this
script emits PGFPlots sources and compiles them with pdflatex. The numeric values are
read from the result JSON files where practical and hard-coded only for cells that are
already summarized in the paper from task-specific pilots.
"""
import json
import os
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.abspath(os.path.join(BASE, "..", "..", "paper_latex", "papers", "latent_recovery"))


def j(path):
    with open(os.path.join(BASE, path)) as fh:
        return json.load(fh)


def rate(summary, point, key):
    c = summary[point]
    return c.get(key, 0) / c["n"]


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def compile_tex(name):
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"{name}.tex"],
        cwd=PAPER,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_png(name):
    subprocess.run(
        [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pngalpha",
            "-r170",
            f"-sOutputFile={name}.png",
            f"{name}.pdf",
        ],
        cwd=PAPER,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def dose_response():
    s7 = {
        1: j("results/validated_summary_negstep.json"),
        2: j("results/validated_summary_neghop2.json"),
        3: j("results/validated_summary_neghop3.json"),
        4: j("results/validated_summary_neghop4.json"),
        5: j("results/validated_summary_neghop5.json"),
    }
    s32 = {
        1: j("results_32b/validated_summary_negstep.json"),
        2: j("results_32b/validated_summary_neghop2.json"),
        3: j("results_32b/validated_summary_neghop3.json"),
    }
    sf = j("results/validated_summary_falsehood.json")
    doubt7 = " ".join(f"({k},{rate(v, 'early', 'acknowledged'):.3f})" for k, v in s7.items())
    rec7 = " ".join(f"({k},{rate(v, 'early', 'valid_rederivation'):.3f})" for k, v in s7.items())
    doubt32 = " ".join(f"({k},{rate(v, 'early', 'acknowledged'):.3f})" for k, v in s32.items())
    rec32 = " ".join(f"({k},{rate(v, 'early', 'valid_rederivation'):.3f})" for k, v in s32.items())
    gdoubt = rate(sf, "early", "acknowledged")
    grec = rate(sf, "early", "valid_rederivation")
    text = rf"""\documentclass[tikz,border=2pt]{{standalone}}
\usepackage{{pgfplots}}
\usetikzlibrary{{calc}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
  name=doubt,width=7.2cm,height=4.4cm,title={{Verbalized doubt}},
  ymin=0,ymax=.46,xmin=.8,xmax=6.25,xtick={{1,2,3,4,5,6}},
  xticklabels={{1,2,3,4,5,$\infty$}},
  xlabel={{distance to contradicting evidence ($k$ hops)}},ylabel={{rate}},
  legend pos=outer north east,grid=both,grid style={{gray!12}},
  tick label style={{font=\small}},label style={{font=\small}},title style={{font=\large}},
]
\addplot+[blue,mark=* ,thick] coordinates {{{doubt7}}};
\addlegendentry{{7B}}
\addplot+[red!70!black,mark=square*,thick,dashed] coordinates {{{doubt32}}};
\addlegendentry{{32B}}
\addplot+[only marks,mark=star,mark size=4pt,black] coordinates {{(6,{gdoubt:.3f})}};
\node[align=center,font=\scriptsize] at (axis cs:5.55,.09) {{global-only\\off-path lie}};
\end{{axis}}
\begin{{axis}}[
  at={{($(doubt.east)+(2.0cm,0)$)}},anchor=west,
  width=7.2cm,height=4.4cm,title={{Closure-valid completion}},
  ymin=.24,ymax=.82,xmin=.8,xmax=6.25,xtick={{1,2,3,4,5,6}},
  xticklabels={{1,2,3,4,5,$\infty$}},
  xlabel={{distance to contradicting evidence ($k$ hops)}},ylabel={{rate}},
  legend pos=outer north east,grid=both,grid style={{gray!12}},
  tick label style={{font=\small}},label style={{font=\small}},title style={{font=\large}},
]
\addplot+[blue,mark=* ,thick] coordinates {{{rec7}}};
\addlegendentry{{7B}}
\addplot+[red!70!black,mark=square*,thick,dashed] coordinates {{{rec32}}};
\addlegendentry{{32B}}
\addplot+[only marks,mark=star,mark size=4pt,black] coordinates {{(6,{grec:.3f})}};
\node[align=center,font=\scriptsize] at (axis cs:5.45,.48) {{global-only\\off-path lie}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    write(os.path.join(PAPER, "fig_dose_response.tex"), text)
    compile_tex("fig_dose_response")
    render_png("fig_dose_response")


def regime_map():
    sf = j("results/EXPA_GLOBAL_EXPANSION/summary_tables.json")["metrics"]["by_condition_position"]["global_falsehood|mid"]
    gsm = j("results/gsm8k/summary.json")
    arith = j("results/arith/summary.json")["mid"]
    pr_rec = sf["valid_rederivation"]["rate"]
    pr_abs = sf["poisoned"]["rate"]
    text = rf"""\documentclass[tikz,border=2pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
  ybar,width=10.8cm,height=5.6cm,bar width=16pt,
  ymin=0,ymax=1.08,ylabel={{rate}},
  symbolic x coords={{PrOntoQA,GSM8K,Chained}},
  xtick=data,
  xticklabels={{PrOntoQA\\(redundant),GSM8K\\(math),Chained arith.\\(no redundancy)}},
  xticklabel style={{align=center,font=\scriptsize,text width=2.7cm}},
  tick label style={{font=\small}},label style={{font=\small}},
  title={{Task derivation structure sets the price of using a planted error}},
  title style={{font=\small}},
  legend style={{at={{(0.02,0.98)}},anchor=north west,draw=none,font=\small}},
  nodes near coords={{\pgfmathprintnumber[fixed,zerofill,precision=3]{{\pgfplotspointmeta}}}},
  nodes near coords style={{font=\scriptsize,text=black}},
  point meta=y,
  enlarge x limits=.26,grid=both,grid style={{gray!12}},
]
\addplot+[black,fill=black!15,draw=black,
  nodes near coords style={{font=\scriptsize,text=black,anchor=south,xshift=-4pt,yshift=1pt}}
] coordinates {{
  (PrOntoQA,{pr_rec:.3f}) (GSM8K,{gsm['recovered_rate']:.3f}) (Chained,{arith['recovered_rate']:.3f})
}};
\addlegendentry{{closure-valid}}
\addplot+[black,fill=black!60,draw=black,
  nodes near coords style={{font=\scriptsize,text=black,anchor=south,xshift=4pt,yshift=1pt}}
] coordinates {{
  (PrOntoQA,{pr_abs:.3f}) (GSM8K,{gsm['poisoned_rate']:.3f}) (Chained,{arith['poisoned_next_step_rate']:.3f})
}};
\addlegendentry{{injection-dependent}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    write(os.path.join(PAPER, "fig_regime_map.tex"), text)
    compile_tex("fig_regime_map")
    render_png("fig_regime_map")


def negstep_mid(resdir):
    d = j(os.path.join(resdir, "validated_summary_negstep.json"))
    c = d["mid"]
    n = c["n"]
    return c.get("acknowledged", 0) / n, c["valid_rederivation"] / n, n


def expa_onehop_mid():
    c = j("results/EXPA_GLOBAL_EXPANSION/summary_tables.json")["metrics"]["by_condition_position"]["one_hop_falsehood|mid"]
    return c["verbalized_doubt"]["rate"], c["valid_rederivation"]["rate"], c["n"]


def verbalization_spectrum():
    r1 = j(os.path.join("results", "r1", "summary.json"))["negstep"]
    pts = [
        ("Qwen-1.5B", *negstep_mid("results_1p5b"), "gray", "0.090,0.505", "west"),
        ("OLMo-2-7B", *negstep_mid("results_olmo"), "green!55!black", "0.090,0.475", "west"),
        ("Llama-3.1-8B", *negstep_mid("results_llama"), "olive", "0.055,0.565", "west"),
        ("Qwen-7B", *expa_onehop_mid(), "black!45", "0.310,0.615", "west"),
        ("Qwen-32B", *negstep_mid("results_32b"), "cyan!70!black", "0.168,0.754", "west"),
        ("R1-Distill-7B", r1["ack_visible"], r1["valid_rederivation"], r1["n"], "red!70!black", "0.655,0.662", "east"),
    ]
    plots = []
    labels = []
    for label, doubt, rec, n, color, labpos, anchor in pts:
        plots.append(
            rf"\addplot+[only marks,mark=*,mark size=2.5pt,draw=black,fill={color}] coordinates {{({doubt:.3f},{rec:.3f})}};"
        )
        labels.append(
            rf"\node[anchor={anchor},font=\scriptsize] at (axis cs:{labpos}) {{{label} (n={n})}};"
        )
    text = rf"""\documentclass[tikz,border=2pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=10.4cm,height=5.8cm,
  xmin=-.035,xmax=.76,ymin=.45,ymax=.79,
  xlabel={{verbalized doubt (mid, 1-hop falsehood)}},
  ylabel={{closure-valid completion}},
  grid=both,grid style={{gray!12}},
  tick label style={{font=\small}},label style={{font=\small}},
  clip=false,
]
{chr(10).join(plots)}
\draw[gray!55,thin] (axis cs:0.018,0.527) -- (axis cs:0.090,0.505);
\draw[gray!55,thin] (axis cs:0.013,0.526) -- (axis cs:0.090,0.475);
\draw[gray!55,thin] (axis cs:0.000,0.533) -- (axis cs:0.055,0.565);
{chr(10).join(labels)}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    write(os.path.join(PAPER, "fig_verbalization_spectrum.tex"), text)
    compile_tex("fig_verbalization_spectrum")
    render_png("fig_verbalization_spectrum")


def main():
    dose_response()
    regime_map()
    verbalization_spectrum()
    print("wrote workshop fig_*.{tex,pdf,png}")


if __name__ == "__main__":
    main()
