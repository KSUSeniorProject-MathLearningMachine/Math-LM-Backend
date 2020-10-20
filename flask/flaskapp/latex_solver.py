from sympy.parsing.latex import parse_latex
from sympy.parsing.latex import LaTeXParsingError
from sympy import latex, simplify

def solve(latex_styled):
    latex_formatted = format_latex(latex_styled)
    try:
        expr = parse_latex(latex_formatted)
    except LaTeXParsingError:
        return "LaTeX parsing error"

    try:
        result = latex(simplify(expr))
    except Exception:
        return "SymPy evaluation error"
    return str(result)

def format_latex(latex_styled):
    return r'{}'.format(latex_styled)