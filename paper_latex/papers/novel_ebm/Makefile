MAIN := main

.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex || true
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).fls $(MAIN).fdb_latexmk
