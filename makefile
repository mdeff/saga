SRC = $(wildcard *.tex)
PDF = $(SRC:.tex=.pdf)

all: $(PDF)
  
%.pdf: %.tex
	@latexmk $<

clean:
	rm -f *.{aux,bbl,blg,fdb_latexmk,fls,log,out}
	rm -f *.{nav,snm,toc}
	rm -f *.{bcf,run.xml}
