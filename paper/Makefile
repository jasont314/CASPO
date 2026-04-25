TEX = main.tex
PDF = main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) references.bib
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex main
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

clean:
	rm -f main.aux main.bbl main.blg main.log main.out $(PDF)
