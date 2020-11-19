

DICT={
    's i n': '\\sin',
    'c o s': '\\cos',
    'l i m': '\\lim',
    'l o g': '\\log',
    't a n': '\\tan',
}


def label_to_latex(label):
    try:
        return DICT[label]
    except KeyError:
        return label


def parse(detections):
    """Parse the detections and place them in order"""
    latex = ''

    detections.sort(key=lambda x: x[1][0]) # Sort by startX value

    for (label, (startX, startY), (endX, endY), confidence) in detections:
        latex += label_to_latex(label)

    return latex
