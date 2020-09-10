from sympy.parsing.latex import parse_latex
from sympy import latex

def solve(latex_styled):
    latex_formatted = format_latex(latex_styled)
    expr = parse_latex(latex_formatted)

    return str(latex(expr.doit()))

def format_latex(latex_styled):
    return r'{}'.format(latex_styled)