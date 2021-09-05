import cv2
import numpy as np
import imutils

def show_image(debug_state, image, show_area=1080):
	if debug_state == True:
		print("   Press any key to go to the next")
		cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
		image = imutils.resize(image, show_area)
		cv2.imshow("Display", image)
		cv2.waitKey(0)
