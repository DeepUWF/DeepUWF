import cv2
import numpy as np
from PIL import Image


def circle_crop(img, sigmaX=10, scale=1.0):   
    """
    Create circular crop around image centre 
    Scale(0~1) is a percentage of original image
    """    
    img = np.asarray(img)
    height, width, depth = img.shape    

    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y)) * scale
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    # bitwise_and 来裁剪原始图像，得到一个圆形图像
    img = cv2.bitwise_and(img, img, mask=circle_img)
    #img = crop_image_from_gray(img)
    #img= cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return Image.fromarray(img)