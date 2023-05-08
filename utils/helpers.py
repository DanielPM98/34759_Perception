import numpy as np
import cv2


def get_depth(imgL, imgR):
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # focal_length = K_02[0][0]
    focal_length = 956.9475
    
    # b = np.linalg.norm(t_02-t_03)
    b = np.linalg.norm(np.array([0.059896, -0.001367835, 0.004637624 ])-np.array([-0.4756270, 0.005296617, -0.005437198]))
    
    stereo = cv2.StereoBM_create()
    stereo.setMinDisparity(4)
    stereo.setNumDisparities(128)
    stereo.setBlockSize(21)
    stereo.setSpeckleRange(16)
    stereo.setSpeckleWindowSize(45)
    
    disparity = stereo.compute(imgL,imgR)
    disparity[disparity <=0] = 1e-5
   
    depth = (b * focal_length)  / disparity
        
    return depth

def draw_text(img, text1, text2,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=0.3,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0,0,255),
          alpha = 0 
          ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text2, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + 2*text_h), text_color_bg, -1)
    cv2.putText(img, text1, (x,  y + text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img, text2, (x,  y + 2*text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if alpha > 0:
        image_new = cv2.addWeighted(img, alpha, img, 1 - alpha, 0)
    else:
        image_new = img

    return image_new
