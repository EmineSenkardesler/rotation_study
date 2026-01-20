# Paper: Crop Rotation Patterns in the US Corn Belt

LaTeX manuscript for the crop rotation study.

## Files

| File | Description |
|------|-------------|
| `main.tex` | Main LaTeX document |
| `references.bib` | BibTeX bibliography |
| `Makefile` | Build automation |
| `figures/` | Publication figures (PNG) |

## Building the PDF

### Option 1: Using Make

```bash
cd /home/emine2/rotation_study/paper

# Build PDF (runs pdflatex + bibtex + pdflatex + pdflatex)
make

# Clean auxiliary files
make clean

# Remove everything including PDF
make distclean

# Open PDF in viewer
make view
```

### Option 2: Manual compilation

```bash
cd /home/emine2/rotation_study/paper

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 3: Using latexmk (if installed)

```bash
latexmk -pdf main.tex
```

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages (included in most distributions):
  - `amsmath`, `amssymb` (math)
  - `graphicx`, `float`, `subcaption` (figures)
  - `booktabs`, `tabularx` (tables)
  - `natbib` (citations)
  - `hyperref` (links)
  - `geometry`, `setspace` (formatting)

## Paper Structure

1. **Abstract** - Summary of methods and findings
2. **Introduction** - Motivation and contributions
3. **Background** - Literature on rotation benefits and economics
4. **Data and Study Area** - CDL description, 8 Corn Belt states
5. **Methods** - Markov chain framework, estimation
6. **Results** - Transition matrices, trends, state comparisons
7. **Discussion** - Interpretation, limitations, policy implications
8. **Conclusion** - Key findings summary
9. **References** - BibTeX bibliography
10. **Appendix** - Additional figures

## Figures

| Figure | Content |
|--------|---------|
| Fig 1 | Transition probability heatmap |
| Fig 2 | Time trends (4 panels) |
| Fig 3 | State crop comparison |
| Fig 4 | Rotation vs continuous rates |
| Fig 5 | Crop area trends (Appendix) |
| Fig 6 | Rotation flow diagram (Appendix) |
