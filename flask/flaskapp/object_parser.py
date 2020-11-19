def parse(detections):
    """Parse the detections and place them in order"""
    latex = r""

    detections.sort(key=lambda x: x[0][0][0])  # Sort by startX value

    for (((startX, startY), (endX, endY)), label, confidence) in detections:
        latex += r"{}".format(label)

    return latex
