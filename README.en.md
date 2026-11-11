# Guide to Econometrics and Time Series

🇧🇷 [Versão em português](README.md)

Study material with original explanations, numerically checked experiments and
worked exercises. It covers: conditional expectation, asymptotics, OLS, omitted
variables and measurement error, instrumental variables, panel data, ARMA/ARIMA,
forecasting, unit roots and cointegration, VAR, ARCH/GARCH and machine learning
for time series.

**PDF (English):** [`en/pdf/Guide_to_Econometrics_and_Time_Series.pdf`](en/pdf/Guide_to_Econometrics_and_Time_Series.pdf)
**PDF (Portuguese):** [`pt/pdf/Guia_de_Econometria_e_Series_Temporais.pdf`](pt/pdf/Guia_de_Econometria_e_Series_Temporais.pdf)

## Important notice

- The text was produced with the help of artificial intelligence (Claude, by Anthropic).
- Formulas and numbers were checked by simulation (folder `code/`) and the text was
  reviewed twice, but it has **not been reviewed by a subject-matter expert**. It may contain errors.
- This guide is **not a translation** of any book. The books mentioned are only
  suggestions for further reading and are not included here.
- If you find an error, please open an issue stating the chapter and the passage.

## Structure

```
pt/pdf/    Portuguese PDF              en/pdf/    English PDF
pt/tex/    LaTeX source and figures    en/tex/    LaTeX source and figures
code/      Python scripts that generate the figures and check the results
```

## How to reproduce

Simulations and figures (Python 3.10+):

```bash
pip install -r requirements.txt
cd code
python sims.py en       # or: python sims.py pt
python checks3.py en    # or: python checks3.py pt
python checks2.py
python checks4.py
```

Figures are written to `en/tex/fig/` or `pt/tex/fig/`, depending on the chosen language.
Numbers depend on fixed seeds but may vary slightly across library versions.

Build the PDF (requires XeLaTeX, Latin Modern fonts and the `polyglossia` package):

```bash
cd en/tex   # or pt/tex
xelatex guia.tex
xelatex guia.tex   # second pass for table of contents and references
```

## License

CC BY 4.0. See `LICENSE.md`.
