#!/usr/bin/python3
import sympy
import sympy.physics
from sympy import symbols
from sympy.vector import CoordSys3D

def strToSympy(str):
    return sympy.parsing.sympy_parser.parse_expr(str, transformations=(sympy.parsing.sympy_parser.standard_transformations + (sympy.parsing.sympy_parser.convert_equals_signs,)))
functions = {
    "\\sin": sympy.sin,
    "\\cos": sympy.cos,
    "\\tan": sympy.tan,
    "\\cot": sympy.cot,
    "\\arcsin": sympy.asin,
    "\\arccos": sympy.acos,
    "\\arctan": sympy.atan,
    "\\arccot": sympy.acot,
    "\\sinh": sympy.sinh,
    "\\cosh": sympy.cosh,
    "\\tanh": sympy.tanh,
    "\\coth": sympy.coth,
    "\\sec": sympy.sec,
    "\\csc": sympy.csc,
    "\\exp": sympy.exp,
    "\\ln": sympy.ln
}
comparators = {
    "=": sympy.Eq,
    "<": sympy.StrictLessThan,
    ">": sympy.StrictGreaterThan,
    "\\leq": sympy.LessThan,
    "\\geq": sympy.GreaterThan
}
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
def printPos(latex, pos):
    #print("position:", pos, latex[pos:min(pos+5,len(latex))])
    #print("position:", latex[:pos] + color.UNDERLINE + color.RED + latex[pos:] + color.END)
    print("position:", latex[:pos] + color.BOLD + "|" + color.END + latex[pos:])
class ConfigL2S:
    def __init__(self, positiveSymbols=[], realSymbols=[], DEBUG=False):
        self.positiveSymbols = positiveSymbols
        self.realSymbols = realSymbols
        self.DEBUG = DEBUG
def createSymbol(str, conf):
    if str in conf.positiveSymbols:
        return symbols(str, positive=True)
    elif str in conf.realSymbols:
        return symbols(str, real=True)
    else:
        return symbols(str)
def jumpToNormal(latex, pos, searchList):
    depthRoundBraces = 0
    depthCurledBraces = 0
    depthBegins = 0
    while pos < len(latex):
        if depthRoundBraces == 0 and depthCurledBraces == 0 and depthBegins == 0 and latex[pos:].startswith(tuple(searchList)):
            break
        if latex[pos] == "(":
            depthRoundBraces += 1
        elif latex[pos] == ")":
            depthRoundBraces -= 1
        elif latex[pos] == "{":
            depthCurledBraces += 1
        elif latex[pos] == "}":
            depthCurledBraces -= 1
        elif latex[pos:pos+6] == "\\begin":
            while latex[pos] != "}":
                pos += 1
            depthBegins += 1
        elif latex[pos:pos+4] == "\\end":
            while latex[pos] != "}":
                pos += 1
            depthBegins -= 1
        pos += 1
    return pos
def findEndOfIndexOrOperand(latex, pos): #Used for a^b, a_b and \frac a b
    #print("Begin of findEndOfIndexOrOperand ", end ='')
    #printPos(latex, pos)
    while pos < len(latex) and latex[pos].isspace():
        pos += 1
    if pos < len(latex) and latex[pos] == "{":
        pos += 1
        depth = 1
        while depth != 0:
            if latex[pos] == "{":
                depth += 1
            elif latex[pos] == "}":
                depth -= 1
            pos += 1
    elif pos < len(latex) and latex[pos] == "\\":
        startpos = pos
        pos += 1
        while pos < len(latex) and latex[pos].isalpha():
            pos += 1
        if latex[startpos:pos] == "\\text":
            pos = findEndOfIndexOrOperand(latex, pos)
    else:
         pos += 1
    #print("End of findEndOfIndexOrOperand ", end ='')
    #printPos(latex, pos)
    return pos
def findEndOfAdd(latex, pos): #Used for a+b and a-b
    roundDepth = 0
    curledDepth = 0
    while pos < len(latex):
        if latex[pos] == "(":
            roundDepth += 1
        elif latex[pos] == ")":
            roundDepth -= 1
        elif latex[pos] == "{":
            curledDepth += 1
        elif latex[pos] == "}":
            curledDepth -= 1
        if roundDepth == 0 and curledDepth == 0 and (latex[pos] in ["+", "-", "=", ">", "<"] or latex[pos:pos+3] in ["\\leq", "\\geq"]):
            break
        pos += 1
    return pos
def findEndOfValue(latex, pos): #Used for a+b, a-b, a/b, \sin a and at the beginning of the main loop
    #print("Begin of findEndOfValue ", end ='')
    #printPos(latex, pos)
    while latex[pos].isspace():
        pos += 1
    if latex[pos:pos+5] == "\\left":
        pos += 5
    if latex[pos] == "(":
        pos += 1
        depth = 1
        while depth != 0:
            if latex[pos] == "(":
                depth += 1
            elif latex[pos] == ")":
                depth -= 1
            pos += 1
    elif latex[pos] == "\\":
        startpos = pos
        pos += 1
        while latex[pos].isalpha():
            pos += 1
        if latex[startpos:pos] == "\\frac":
            pos = findEndOfIndexOrOperand(latex, pos)
            pos = findEndOfIndexOrOperand(latex, pos)
        elif latex[startpos:pos] == "\\bm":
            pos = findEndOfIndexOrOperand(latex, pos)
    elif latex[pos].isdigit():
        while pos < len(latex) and (latex[pos].isdigit() or latex[pos].isspace() or latex[pos]=='.'):
            pos += 1
    else:
        pos += 1
    while pos < len(latex) and latex[pos].isspace():
        pos += 1
    if pos < len(latex) and latex[pos] in ["_", "^"]:
        pos = findEndOfIndexOrOperand(latex, pos+1)
    while pos < len(latex) and latex[pos].isspace():
        pos += 1
    if pos < len(latex) and latex[pos] in ["_", "^"]:
        pos = findEndOfIndexOrOperand(latex, pos+1)
    #print("End of findEndOfValue ", end ='')
    #printPos(latex, pos)
    return pos
def findEndOfRoundBrace(latex, pos): #Like findEndOfValue, but (a+b)^c returns (a+b) instead of (a+b)^c
    if latex[pos] != "(":
        print("you are using findEndOfRoundBrace wrong")
        raise ParsingError
    pos += 1
    depth = 1
    while depth != 0:
        if latex[pos] == "(":
            depth += 1
        elif latex[pos] == ")":
            depth -= 1
        pos += 1
    return pos
    
    #Uncomment the following for another working solution. I just wrote the above the decrese the number of lines of code
    
    #while latex[pos].isspace():
    #    pos += 1
    #if latex[pos] == "(":
    #    pos += 1
    #    depth = 1
    #    while depth != 0:
    #        if latex[pos] == "(":
    #            depth += 1
    #        elif latex[pos] == ")":
    #            depth -= 1
    #        pos += 1
    #elif latex[pos] == "\\":
    #    startpos = pos
    #    pos += 1
    #    while latex[pos].isalpha():
    #        pos += 1
    #    if latex[startpos:pos] == "\\frac":
    #        pos = findEndOfIndexOrOperand(latex, pos)
    #        pos = findEndOfIndexOrOperand(latex, pos)
    #elif latex[pos].isdigit():
    #    while pos < len(latex) and (latex[pos].isdigit() or latex[pos].isspace() or latex[pos] == '.'):
    #        pos += 1
    #else:
    #    pos += 1
    #return pos
def removeComments(latex):
    cleaned = ""
    pos = 0
    while pos < len(latex):
        if latex[pos] == "%":
            while pos < len(latex) and latex[pos] != "\\n":
                pos += 1
            pos += 1
        if pos < len(latex):
            cleaned += latex[pos]
        pos += 1
    return cleaned
Sys = CoordSys3D("Sys")
O = Sys.origin
class ParsingError(Exception):
    pass
def latex2sympy(latex, conf, indent=0):
    if conf.DEBUG:
        print("\t"*indent + color.BOLD + "input: " + latex + color.END)
    latex = removeComments(latex)
    ret = 1
    empty = True
    pos = 0
    while pos < len(latex):
        if conf.DEBUG:
            print("\t"*indent + "loop ret: " + str(ret))
            print("\t"*indent + latex[:pos] + color.BOLD + "|" + color.END + latex[pos:])
        possym = findEndOfIndexOrOperand(latex, pos)
        plannedMul = None #Don't do stuff like ret = sympy.Mul(ret, symbols("\\alpha")), do plannedMul = symbols("\\alpha"). If plannedMul is set, this skript will check for trailing _ and ^ 
        if latex[pos].isspace(): #Handling of spaces
            pos += 1
        elif latex[pos].isdigit(): #Handling of numbers
            startpos = pos
            hasPoint = False
            while pos < len(latex) and (latex[pos].isdigit() or latex[pos].isspace() or latex[pos]=='.'):
                if latex[pos] == ".":
                    hasPoint = True
                pos += 1
            if hasPoint:
                plannedMul = float(latex[startpos:pos]) #Were doing plannedMul = ... instead of ret = sympy.Mul(ret, ...) because of latex code like 3^b
            else:
                plannedMul = int(latex[startpos:pos]) #Were doing plannedMul = ... instead of ret = sympy.Mul(ret, ...) because of latex code like 3^b
        elif latex[pos].isalpha(): #Handling of single symbols
            plannedMul = createSymbol(latex[pos], conf) #Were doing plannedMul = ... instead of ret = sympy.Mul(ret, ...) because of latex code like a^b
            pos += 1
        elif latex[pos] in ["+", "-"]:#Handling of additions and subtractions 
            startpos = pos
            pos = findEndOfAdd(latex, pos+1)
            term = latex2sympy(latex[startpos+1:pos], conf, indent+1)
            if latex[startpos] == "+" and empty:
                ret = term
            elif latex[startpos] == "+":
                ret = sympy.Add(ret, term, evaluate=False)
            elif latex[startpos] == "-" and empty:
                ret = sympy.Mul(term, -1, evaluate=False)
            elif latex[startpos] == "-":
                ret = sympy.Add(ret, sympy.Mul(term, -1, evaluate=False), evaluate=False)
            empty = False
        elif latex[pos] == "/":#Handling of divisions
            startpos = pos
            pos = findEndOfValue(latex, pos+1)
            term = latex2sympy(latex[startpos+1:pos], conf, indent+1)
            ret = sympy.Mul(ret, sympy.Pow(term, -1, evaluate=False), evaluate=False)
            empty = False
        elif latex[pos:possym] in tuple(comparators.keys()):
            for c in comparators:
                if latex[pos:possym].startswith(c):
                    comp = c
            if ret.func in comparators.values():
                ret = ret & comparators[comp](ret.args[-1], latex2sympy(latex[possym:], conf, indent+1),evaluate=False)
            elif ret.func == sympy.And:
                ret = ret & comparators[comp](ret.args[-1].args[-1], latex2sympy(latex[possym:], conf, indent+1),evaluate=False)
            else:
                ret = comparators[comp](ret, latex2sympy(latex[possym:], conf, indent+1), evaluate=False)
            #empty = False
            break
        elif latex[pos] == "\\": #Handling of things like \\alpha and \\frac and \\sin
            startpos = pos
            pos += 1
            while pos < len(latex) and latex[pos].isalpha():
                pos += 1
            if latex[startpos:pos] == "\\frac":
                startOfFirstOperand = pos
                endOfFirstOperand = findEndOfIndexOrOperand(latex, pos)
                endOfSecondOperand = findEndOfIndexOrOperand(latex, endOfFirstOperand)
                pos = endOfSecondOperand
                ret = sympy.Mul(ret, latex2sympy(latex[startOfFirstOperand:endOfFirstOperand], conf, indent+1), evaluate=False)
                term = sympy.Pow(latex2sympy(latex[endOfFirstOperand:endOfSecondOperand], conf, indent+1), -1, evaluate=False)
                ret = sympy.Mul(ret, term, evaluate=False)
                empty = False
            elif latex[startpos:pos] == "\\mapsfrom":
                ret = sympy.codegen.ast.Assignment(ret, latex2sympy(latex[pos:],conf, indent+1))
                pos = len(latex)
            elif latex[startpos:pos] == "\\sqrt":
                #print("========SQRT======")
                while pos < len(latex) and latex[pos].isspace():
                    pos += 1
                root = 2
                if latex[pos] == "[":
                    pos1 = pos
                    while latex[pos] != "]":
                        pos += 1
                    root = latex2sympy(latex[pos1+1:pos], conf, indent+1)
                    pos += 1
                pos1 = pos
                #printPos(latex, pos)
                pos = findEndOfIndexOrOperand(latex, pos)
                #printPos(latex, pos)
                ret = sympy.Mul(ret, sympy.Pow( latex2sympy(latex[pos1:pos], conf, indent+1) ,1/root,evaluate=False), evaluate=False)
            elif latex[startpos:pos] == "\\exp":
                pos1 = pos
                pos = findEndOfValue(latex, pos)
                ret = sympy.Mul(ret, sympy.Pow(2.7182818284590452353602874713526624977572470937, latex2sympy(latex[pos1:pos], conf, indent+1), evaluate=False), evaluate=False)
            elif latex[startpos:pos] == "\\operatorname":
                pos1 = pos
                pos2 = findEndOfIndexOrOperand(latex, pos)
                pos = findEndOfValue(latex, pos2)
                ret = sympy.Mul(ret, sympy.Function("codegen_operatorname_" + latex[pos1+1:pos2-1])(latex2sympy(latex[pos2:pos],conf,indent+1)), evaluate=False)
            elif latex[startpos:pos] == "\\bm":
                #printPos(latex, pos)
                while latex[pos].isspace():
                    pos += 1
                pos1 = pos
                pos = findEndOfIndexOrOperand(latex, pos)
                if latex[pos1] == "{":
                    name = latex[pos1+1:pos-1]
                else:
                    name = latex[pos1:pos]
                #A = O.locate_new(name, symbols(name+"_x")*Sys.i+ symbols(name+"_y")*Sys.j+ symbols(name+"_z")*Sys.k)
                plannedMul = createSymbol("vec_" + name, conf)
                #printPos(latex, pos)
            elif latex[startpos:pos] in ["\\left", "\\right", "\\cdot"]:
                pass
            elif latex[startpos:pos] == "\\begin":
                pos1 = findEndOfIndexOrOperand(latex, pos)
                if latex[pos:pos1] == "{cases}":
                    start = pos1
                    posAnd = []
                    posNewline = []
                    while True:
                        pos = jumpToNormal(latex,pos,["&", "\\end"])
                        if latex[pos] != "&":
                            break
                        posAnd.append(pos)
                        pos = jumpToNormal(latex,pos,["\\\\", "\\end"])
                        posNewline.append(pos)
                        if latex[pos:pos+2] != "\\\\":
                            break
                    #print("==================")
                    #for i in posAnd:
                    #    printPos(latex, i)
                    #print("------------------")
                    #for i in posNewline:
                    #    printPos(latex, i)
                    #print("==================")
                    piecewise = []
                    for i in range(0,len(posAnd)):
                        if i == 0:
                            sp = start
                        else:
                            sp = posNewline[i-1]+2
                        #print(latex[posAnd[i]+1:posNewline[i]])
                        #print(latex[sp:posAnd[i]])
                        if latex[posAnd[i]+1:posNewline[i]] == "\\text{sonst}":
                            piecewise.append((latex2sympy(latex[sp:posAnd[i]], conf, indent+1),1))
                        else:
                            piecewise.append((latex2sympy(latex[sp:posAnd[i]], conf, indent+1),latex2sympy(latex[posAnd[i]+1:posNewline[i]], conf, indent+1)))
                    ret = sympy.Mul(ret, sympy.Piecewise(*piecewise), evaluate=False)
                    empty = False
                    pos = posNewline[-1]
                    while latex[pos] != "}":
                        pos += 1
                    pos += 1
                else:
                    print("unknown begin")
                    raise ParsingError
            else:
                plannedMul = createSymbol(latex[startpos:pos], conf)
        elif latex[pos] == "(":
            end = findEndOfRoundBrace(latex, pos)
            plannedMul = latex2sympy(latex[pos+1:end-1], conf, indent+1)
            pos = end
        elif latex[pos] in ["{", "}", "*"]: #Handling of useless stuff. This should only detect braces if latex is something like {a+b} + c.
            pos += 1
        else:
            print("Unknown symbol:")
            print(latex[pos])
            raise ParsingError
            
        plannedExp = None
        if plannedMul is not None: #If plannedMul is set, this part checks for _ and ^
            while pos < len(latex) and latex[pos].isspace():
                pos += 1
            middlepos = pos
            if pos < len(latex) and latex[pos] == "^":
                pos = findEndOfIndexOrOperand(latex, pos+1)
                plannedExp = latex2sympy(latex[middlepos+1:pos], conf, indent+1)
            elif pos < len(latex) and latex[pos] == "_":
                pos = findEndOfIndexOrOperand(latex, pos+1)
                plannedMul = createSymbol(str(plannedMul)+latex[middlepos:pos], conf)
        if plannedMul is not None: #We need to do this twice, because a_b^2 and a^2_b is valid latex
            while pos < len(latex) and latex[pos].isspace():
                pos += 1
            middlepos = pos
            if pos < len(latex) and latex[pos] == "^":
                pos = findEndOfIndexOrOperand(latex, pos+1)
                plannedMul = sympy.Pow(plannedMul, latex2sympy(latex[middlepos+1:pos], conf, indent+1), evaluate=False)
            elif pos < len(latex) and latex[pos] == "_":
                pos = findEndOfIndexOrOperand(latex, pos+1)
                plannedMul = createSymbol(str(plannedMul)+latex[middlepos:pos], conf)
            if plannedExp is None:
                ret = sympy.Mul(ret, plannedMul, evaluate=False)
            else:
                ret = sympy.Mul(ret, sympy.Pow(plannedMul, plannedExp, evaluate=False), evaluate=False)
            empty = False
    if conf.DEBUG:
        print("\t"*indent + color.BOLD + "return: "+ str(ret) + color.END)
    return ret
