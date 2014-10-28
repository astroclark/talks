default: cc_oct-2014

cc_oct-2014:
	rm -f cc_oct-2014.aux cc_oct-2014.bbl
	pdflatex cc_oct-2014
	bibtex cc_oct-2014
	pdflatex cc_oct-2014
	pdflatex cc_oct-2014
	#bibtex cc_oct-2014
	#latex cc_oct-2014
	#latex cc_oct-2014
	#dvipdf cc_oct-2014
	
	
clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.lot *.lof *.toc *.bbl *.blg *.out *nav *swp *snm *blx.bib
	
