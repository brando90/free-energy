# =====================================================================
# Build the Free Energy paper.
# =====================================================================
#
# Targets:
#   make           — build main.pdf using latexmk (recommended)
#   make quick     — single pdflatex pass (no bib resolution)
#   make full      — pdflatex + bibtex + pdflatex + pdflatex (no latexmk)
#   make clean     — remove build artefacts
#   make distclean — also remove the produced PDF
# =====================================================================

ENGINE  ?= pdflatex
MAIN    := main
BIB     := refs

.PHONY: all quick full clean distclean watch

all:
	latexmk -$(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex

quick:
	$(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex

full:
	$(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex
	bibtex $(MAIN)
	$(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex
	$(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex

watch:
	latexmk -pvc -$(ENGINE) -interaction=nonstopmode -file-line-error $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex || true
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out $(MAIN).toc \
	      $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz \
	      *.aux

distclean: clean
	rm -f $(MAIN).pdf
