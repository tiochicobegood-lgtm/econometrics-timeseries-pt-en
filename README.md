# Guia de Econometria e Séries Temporais

   [English version / Versão em inglês](README.en.md)

Material de estudo em português com explicações originais, experimentos numéricos
conferidos e exercícios resolvidos. Cobre: esperança condicional, assintótica, MQO,
variáveis omitidas e erro de medida, variáveis instrumentais, dados em painel,
ARMA/ARIMA, previsão, raízes unitárias e cointegração, VAR, ARCH/GARCH e
aprendizado de máquina para séries temporais.

**PDF (português):** [`pt/pdf/Guia_de_Econometria_e_Series_Temporais.pdf`](pt/pdf/Guia_de_Econometria_e_Series_Temporais.pdf)
**PDF (inglês):** [`en/pdf/Guide_to_Econometrics_and_Time_Series.pdf`](en/pdf/Guide_to_Econometrics_and_Time_Series.pdf)

## Aviso importante

- O texto foi elaborado com auxílio de inteligência artificial (Claude, da Anthropic).
- As fórmulas e os números foram conferidos por simulação (pasta `code/`) e o texto
  foi revisado duas vezes, mas **não passou por revisão de um especialista**. Pode haver erros.
- Este guia **não é uma tradução** de nenhum livro. Os livros mencionados são apenas
  sugestões de leitura aprofundada e não estão incluídos aqui.
- Se encontrar um erro, abra uma *issue* indicando o capítulo e o trecho.

## Estrutura

```
pt/pdf/    PDF em português            en/pdf/    PDF em inglês
pt/tex/    Fonte LaTeX e figuras (PT)  en/tex/    Fonte LaTeX e figuras (EN)
code/      Scripts Python que geram as figuras e conferem os resultados
```

## Como reproduzir

Simulações e figuras (Python 3.10+):

```bash
pip install -r requirements.txt
cd code
python sims.py pt       # ou: python sims.py en
python checks3.py pt    # ou: python checks3.py en
python checks2.py
python checks4.py
```

As figuras são gravadas em `pt/tex/fig/` ou `en/tex/fig/`, conforme o idioma escolhido.
Os números dependem de sementes fixas, mas podem variar levemente entre versões das bibliotecas.

Compilar o PDF (requer XeLaTeX, fontes Latin Modern e o pacote `polyglossia`):

```bash
cd pt/tex   # ou en/tex
xelatex guia.tex
xelatex guia.tex   # segunda passada para sumário e referências
```

## Licença

CC BY 4.0. Veja `LICENSE.md`.
