import cv2

def draw_boxes(overlay, bbox, color=(0,0,255), thicknes = 2, alpha=0):

    assert alpha >= 0, 'Alpha value cannot be lower than 0'

    for i, box in enumerate(bbox):
        print(box)
        x, y, w, h = box.box
        cv2.rectangle(overlay, (int(x), int(y)), (int(w), int(h)), color, thicknes)
        if alpha > 0:
            image_new = cv2.addWeighted(overlay, alpha, overlay, 1 - alpha,0, 0)


    return image_new
