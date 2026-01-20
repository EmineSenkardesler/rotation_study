# Makefile for LaTeX paper compilation
#
# Usage:
#   make          - Build the PDF
#   make clean    - Remove auxiliary files
#   make distclean - Remove all generated files including PDF
#   make view     - Open PDF in default viewer

# Main file (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Output
PDF = $(MAIN).pdf

# Auxiliary files to clean
AUXFILES = *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean distclean view

# Default target: build PDF
all: $(PDF)

# Build PDF (run pdflatex -> bibtex -> pdflatex -> pdflatex)
$(PDF): $(MAIN).tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	@echo "=== BibTeX pass ==="
	$(BIBTEX) $(MAIN)
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	@echo "=== Build complete: $(PDF) ==="

# Clean auxiliary files
clean:
	@echo "Removing auxiliary files..."
	rm -f $(AUXFILES)
	@echo "Done."

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "Done."

# Open PDF in default viewer
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "Cannot find PDF viewer. PDF is at: $(PDF)"; \
	fi

# Quick build (single pass, no bibliography)
quick:
	$(LATEX) $(LATEXFLAGS) $(MAIN)
