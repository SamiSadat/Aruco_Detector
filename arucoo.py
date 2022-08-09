import cv2
# from object_detector import *
import numpy as np
import cv2


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours



img = cv2.imread("phone_aruco_marker_2.jpg")
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
corners, id, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
print(id)

detector = HomogeneousBgDetector()
# aruco_perimeter = cv2.arcLength(corners[0], True)


contours = detector.detect_objects(img)

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.putText(img, "id : {}".format(id[0][0]), (int(x - 100), int(y - 50)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(img, str(x), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(img, str(y), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)