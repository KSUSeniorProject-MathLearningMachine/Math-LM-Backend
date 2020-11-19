from sympy.parsing.latex import parse_latex
from sympy.parsing.latex import LaTeXParsingError
from sympy import latex, simplify


def solve(latex_styled):
    latex_formatted = format_latex(latex_styled)

    try:
        expr = parse_latex(latex_formatted)

        result = simplify(expr)
        result = latex(result)
        result = str(result)

        return result
    except LaTeXParsingError:
        return "LaTeX parsing error"
    except Exception:
        return "SymPy evaluation error"


def format_latex(latex_styled):
    return r"{}".format(latex_styled)