def parse(detections):
    """Parse the detections and place them in order"""
    latex = ""
    for label in detections:
        latex += label

    latex = latex.replace('T', '+')
    latex = latex.replace('t', '+')
    latex = latex.replace('--', '=')

    ltx = r""

    for label in latex:
        ltx += r"{}".format(label)

    return latex
