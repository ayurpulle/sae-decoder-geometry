"""Convert the write-up doc (XML export from the docs connector) to LaTeX.

Usage: python paper/doc2tex.py DOC_XML OUT_TEX
Figures are mapped from the doc's media blobs to local PNGs under results/ (see FIGURES).
"""
import re
import sys
import xml.etree.ElementTree as ET

# doc media blob -> local figure (matched on the doc's alt text)
FIGURES = {
    "blob/ad23c7bc-24d6": "results/phase1/slope_vs_l0_L12.png",
    "blob/ddf82f36-2f9e": "results/phase1/slope_vs_layer_matched_l0.png",
    "blob/a39830ef-8f46": "results/test1/sigma_vs_decoder_layers.png",
    "blob/3fd5b536-0eb0": "results/test1/gamma_l0_L12.png",
    "blob/4e65aa08-f19d": "results/gpu/gpu_results.png",
}

MATH = {  # unicode -> inline-math LaTeX
    "γ": r"\gamma", "Σ": r"\Sigma", "α": r"\alpha", "μ": r"\mu", "ρ": r"\rho", "λ": r"\lambda",
    "≈": r"\approx", "≤": r"\le", "≥": r"\ge", "±": r"\pm", "×": r"\times", "→": r"\to",
    "∝": r"\propto", "‖": r"\|", "·": r"\cdot", "−": "-", "′": "'",
    "₀": "_0", "₁": "_1", "₂": "_2", "ᵢ": "_i", "²": "^2", "~": r"\sim",
}
TEXT = {"–": "--", "—": "---", "%": r"\%", "&": r"\&", "$": r"\$", "#": r"\#",
        "{": r"\{", "}": r"\}", "_": r"\_", "^": r"\^{}", "\\": r"\textbackslash{}"}
O, C = "\x00", "\x01"  # protected-math sentinels


def math_inner(s):
    s = s.replace("–", r"\textrm{--}")
    return "".join(MATH.get(ch, ch) for ch in s)


def protect(s):
    """Turn caret/underscore notation and the other math-like spans into protected math."""
    s = s.replace("⁻¹", O + "^{-1}" + C)
    s = re.sub(r"([A-Za-zΣλ0-9]+ᵢ?)\^([−\-]?[A-Za-zγ0-9.]+(?:–[0-9.]+)?)",
               lambda m: O + math_inner(m.group(1)) + "^{" + math_inner(m.group(2)) + "}" + C, s)
    s = s.replace("λ_Σ", O + r"\lambda_\Sigma" + C)
    s = s.replace("log_d(d/k)", O + r"\log_d(d/k)" + C)
    s = s.replace("k_aux", O + r"k_{\mathrm{aux}}" + C)
    sub = {"₀": "0", "₁": "1", "₂": "2", "ᵢ": "i"}
    s = re.sub(r"[₀₁₂ᵢ]{2,}", lambda m: O + "_{" + "".join(sub[c] for c in m.group(0)) + "}" + C, s)
    return s


def tex_text(s):
    s = protect(s)
    out, inmath = [], False
    for ch in s:
        if ch == O:
            inmath = True; out.append("$"); continue
        if ch == C:
            inmath = False; out.append("$"); continue
        if inmath:
            out.append(ch); continue
        if ch in MATH:
            out.append("$" + MATH[ch] + "$")
        elif ch in TEXT:
            out.append(TEXT[ch])
        else:
            out.append(ch)
    r = "".join(out)
    r = re.sub(r"(?<!\\)\$\$", "", r)          # join adjacent inline maths
    r = re.sub(r"\$\^\{-\}\$", "$^{-}$", r)
    return r


def inline(el):
    """LaTeX for an element's inline content."""
    parts = [tex_text(el.text or "")]
    for ch in el:
        tag = ch.tag
        if tag == "text":
            parts.append(inline(ch))
        elif tag == "bold":
            parts.append(r"\textbf{" + inline(ch) + "}")
        elif tag in ("italic", "em"):
            parts.append(r"\emph{" + inline(ch) + "}")
        elif tag == "code":
            parts.append(r"\path{" + "".join(ch.itertext()) + "}")
        elif tag == "link":
            parts.append(r"\href{" + ch.get("href").replace("%", r"\%").replace("#", r"\#") + "}{" + inline(ch) + "}")
        elif tag == "date":
            parts.append(ch.get("value"))
        elif tag == "mention":
            parts.append(tex_text(ch.get("name")))
        else:
            parts.append(inline(ch))
        parts.append(tex_text(ch.tail or ""))
    return "".join(parts)


def cell_tex(cell):
    return r" \newline ".join(inline(p) for p in cell if p.tag == "paragraph")


def table(el):
    rows = [r for r in el if r.tag == "row"]
    grid = [[c for c in r if c.tag == "cell"] for r in rows]
    ncol = max(len(r) for r in grid)
    plain = [["".join(c.itertext()) for c in r] + [""] * (ncol - len(r)) for r in grid]
    header = all(c.get("header") == "true" for c in grid[0])
    bodyrows = plain[1:] if header else plain
    maxlen = [max(len(r[j]) for r in bodyrows) for j in range(ncol)]
    numeric = [j > 0 and all(len(r[j]) <= 16 and (re.search(r"\d", r[j]) or r[j].strip() in "—-")
                             for r in bodyrows) for j in range(ncol)]
    body = []
    for i, r in enumerate(grid):
        cells = [cell_tex(c) for c in r] + [""] * (ncol - len(r))
        if i == 0 and header:
            for j, h in enumerate(plain[0]):
                if numeric[j] and len(h) > 12 and " " in h:  # wrap long headers over numeric columns
                    sp = [m.start() for m in re.finditer(" ", h)]
                    cut = min(sp, key=lambda p: abs(p - len(h) / 2))
                    cells[j] = r"\makecell[r]{" + tex_text(h[:cut]) + r" \\ " + tex_text(h[cut + 1:]) + "}"
            cells = [r"\textbf{" + c + "}" if c else c for c in cells]
        body.append(" & ".join(cells) + r" \\")
        if i == 0 and header:
            body.append(r"\midrule")
    if max(maxlen) > 30:  # prose table: wrap the long columns, allow page breaks
        spec = "".join(r">{\raggedright\arraybackslash}X" if m > 30 else ("r" if numeric[j] else "l")
                       for j, m in enumerate(maxlen))
        head = body[:2] if header else []
        rest = body[2:] if header else body
        size = r"\small" if ncol <= 5 else r"\footnotesize\setlength{\tabcolsep}{4pt}"
        return "\n".join(["{" + size, r"\begin{xltabular}{\textwidth}{" + spec + "}", r"\toprule", *head,
                          r"\endhead", r"\bottomrule", r"\endlastfoot", *rest, r"\end{xltabular}}", ""])
    spec = "".join("r" if numeric[j] else "l" for j in range(ncol))
    return "\n".join([r"\begin{center}", r"\begin{adjustbox}{max width=\textwidth}", r"\small",
                      r"\begin{tabular}{" + spec + "}", r"\toprule", *body, r"\bottomrule",
                      r"\end{tabular}", r"\end{adjustbox}", r"\end{center}", ""])


def lst(el):
    env = "enumerate" if el.get("kind") == "ordered" else "itemize"
    out = [r"\begin{" + env + "}"]
    for item in el:
        if item.tag != "listItem":
            continue
        chunks = [block(b) for b in item]
        out.append(r"\item " + "\n".join(c for c in chunks if c).strip())
    out.append(r"\end{" + env + "}")
    return "\n".join(out) + "\n"


def block(el):
    tag = el.tag
    if tag == "paragraph":
        media = [block(m) for m in el.iter("media")]
        if media:  # figures sit inside otherwise-empty paragraphs
            return "".join(media)
        body = inline(el).strip()
        if not body:
            return ""
        h = el.get("heading")
        if h == "2":
            return r"\section{" + body + "}\n"
        if h == "3":
            return r"\subsection{" + body + "}\n"
        return body + "\n"
    if tag == "table":
        return table(el)
    if tag == "list":
        return lst(el)
    if tag == "media":
        path = FIGURES[el.get("blob")]
        return "\n".join([r"\begin{figure}[!htbp]", r"\centering",
                          r"\includegraphics[width=\textwidth]{" + path.split("/", 1)[1] + "}",
                          r"\caption{" + tex_text(el.get("alt")) + ".}", r"\end{figure}", ""])
    if tag == "codeBlock":
        code = "".join(el.itertext()).strip()
        if el.get("language") == "latex":
            return r"\[" + "\n" + code + "\n" + r"\]" + "\n"
        return r"\begin{verbatim}" + "\n" + code + "\n" + r"\end{verbatim}" + "\n"
    return ""


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\graphicspath{{../results/}}
\usepackage{booktabs,xltabular,array,adjustbox,makecell}
\usepackage{enumitem}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}
\setlist{itemsep=0.3em}
\renewcommand{\arraystretch}{1.15}
"""


def main(src, dst):
    root = ET.fromstring(open(src).read())
    blocks = list(root)
    title = inline(blocks[0])
    byline = blocks[1]
    date = byline.find("date").get("value") if byline.find("date") is not None else ""
    author = byline.find("mention").get("name") if byline.find("mention") is not None else ""
    body = [block(b) for b in blocks[2:]]
    # the first body paragraph is the doc's summary: set it as the abstract
    first = next(i for i, b in enumerate(body) if b)
    abstract = body[first].strip()
    rest = body[first + 1:]
    doc = [PREAMBLE, r"\title{" + title + "}", r"\author{" + tex_text(author) + "}",
           r"\date{November 2026}", "", r"\begin{document}", r"\maketitle", "",
           r"\begin{abstract}", abstract, r"\end{abstract}", "", *[b for b in rest if b], r"\end{document}", ""]
    open(dst, "w").write("\n".join(doc))


if __name__ == "__main__":
    main(*sys.argv[1:3])
