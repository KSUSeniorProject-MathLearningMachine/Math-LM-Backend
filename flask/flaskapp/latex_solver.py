from sympy.parsing.latex import parse_latex
from sympy.parsing.latex import LaTeXParsingError
from sympy import latex

def solve(latex_styled):
    latex_formatted = format_latex(latex_styled)
    try:
        expr = parse_latex(latex_formatted)
    except LaTeXParsingError:
        return "LaTeX parsing error"

    try:
        result = latex(expr.doit())
        #This is super generic because sympy evaluation isn't trivial. 
        # We'll probably want to switch away from using doit() in the future, for more control.
    except Exception:
        return "SymPy evaluation error"
    return str(result)

def format_latex(latex_styled):
    return r'{}'.format(latex_styled)