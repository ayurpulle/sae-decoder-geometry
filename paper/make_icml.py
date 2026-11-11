"""Write thesisA_icml.tex: thesisA.tex body under the official ICML 2026 style (icml_style/), for the page-limit check.
  python make_icml.py && TEXINPUTS=icml_style: BSTINPUTS=icml_style: latexmk -pdf -outdir=build thesisA_icml.tex
"""
import re
from pathlib import Path

src = Path(__file__).with_name("thesisA.tex").read_text()
pre, body = src.split(r"\begin{document}", 1)
title = re.search(r"\\title\{(.*?)\}\n", pre).group(1)
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
body = body[body.index("]", body.index(r"\end{abstract}")) + 1:]
body = body.replace(r"\bibliographystyle{plainnat}", r"\bibliographystyle{icml2026}")
macros = "\n".join(l for l in pre.splitlines() if l.startswith((r"\newcommand", r"\graphicspath", r"\DeclareMathOperator")))
head = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{icml2026}
\usepackage{amsmath,amssymb}
""" + macros + r"""
\icmltitlerunning{""" + title + r"""}
\begin{document}
\twocolumn[
  \icmltitle{""" + title + r"""}
  \begin{icmlauthorlist}
    \icmlauthor{Anonymous Authors}{anon}
  \end{icmlauthorlist}
  \icmlaffiliation{anon}{Anonymous Institution}
  \icmlcorrespondingauthor{Anonymous Authors}{anon@example.org}
  \icmlkeywords{sparse autoencoders, interpretability}
  \vskip 0.3in
]
\printAffiliationsAndNotice{}
\begin{abstract}""" + abstract + r"""\end{abstract}
"""
Path(__file__).with_name("thesisA_icml.tex").write_text(head + body)
