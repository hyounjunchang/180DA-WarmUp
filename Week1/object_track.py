import cv2
import numpy as np

cap = cv2.VideoCapture(0)

"""
Code Modified from
https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python
for drawing Boundary rectangles
and
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
for ColorSpaces/Video in open CV

Small decision changes suc as color bounds and boundary size minimum was added
"""

in_HSV = True


while(1):

    # Take each frame
    _, image = cap.read()

    colorspace_img = image

    #since the color space is different, i would need to choose different colors
    #lower_bound = np.array([50, 43, 40]) #direct transation from HSV values, inRange function does not like it likely
    #upper_bound = np.array([255,0, 85])
    lower_bound = np.array([50, 0, 40])
    upper_bound = np.array([255,43, 85])

    if in_HSV:
        # Convert BGR to HSV (optional)
        colorspace_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # define range of Thresholds (in BGR or HSV)
        lower_bound = np.array([110,50,50])
        upper_bound = np.array([130,255,255])

    # Threshold the HSV image to get only Bounded
    mask = cv2.inRange(colorspace_img, lower_bound, upper_bound)
    # Bitwise-AND mask and original image
    masked_image = cv2.bitwise_and(image,image, mask= mask)

    
    # Convert the masked_image (containing filtered pixels) to Grayscale
    original = image.copy()
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 2,255,cv2.THRESH_BINARY)


    # Find contours, obtain bounding box, extract
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        #ignore small boundaries
        if cv2.contourArea(c) < 100:
            continue

        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

    cv2.imshow('Image Detection Based on Color', image)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Original', original)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()