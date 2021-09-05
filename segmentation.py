import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

def show_image(debug_state, image, show_area=1080):
	if debug_state == True:
		print("   Press any key to go to the next")
		cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
		image = imutils.resize(image, show_area)
		cv2.imshow("Display", image)
		cv2.waitKey(0)

def white_background_segmentation(img, debug=False, width_ratio=1080):

	# 1. convert the image to RGB
	print("1. Read_image")
	show_image(debug, img, width_ratio)

	# 2.  Apply Median Blur
	median = cv2.medianBlur(img, 13)
	print("2. Apply median filter")
	show_image(debug, median, width_ratio)

	# 3. get negative image
	img_neg = cv2.bitwise_not(median)
	print("3. Get inverted image")
	show_image(debug, img_neg, width_ratio)

	# 4. Convert the image to gray-scale
	gray = cv2.cvtColor(img_neg, cv2.COLOR_BGR2GRAY)
	print("4. Get grayscale")
	show_image(debug, gray, width_ratio)

	# 5. Apply thresholding
	(thresh, img_thresh) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
	h, w = img_thresh.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	print("flood")
	cv2.floodFill(img_thresh, mask, (0,0), 127)
	show_image(debug, img_thresh, width_ratio)

	(thresh, img_thresh_white) = cv2.threshold(img_thresh, 250, 255, cv2.THRESH_BINARY)
	print("white")
	show_image(debug, img_thresh_white, width_ratio)

	img_thresh_neg = cv2.bitwise_not(img_thresh)
	(thresh, img_thresh_black) = cv2.threshold(img_thresh_neg, 250, 255, cv2.THRESH_BINARY)
	print("black")
	show_image(debug, img_thresh_black, width_ratio)

	img_thresh = img_thresh_white + img_thresh_black
	print("5. Apply thresholding")
	show_image(debug, img_thresh, width_ratio)

	# 6. Find the edges
	edges = cv2.Canny(img_thresh, 100, 200)
	print("6. Get edges")
	show_image(debug, edges, width_ratio)

	contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# 7. Find the convex hull from contours and draw it on the original image.
	convex_hull = edges
	for i in range(len(contours)):
		hull = cv2.convexHull(contours[i])
	cv2.drawContours(convex_hull, [hull], -1, (255, 0, 0), -1)
	print("7. Get convex hull")
	show_image(debug, convex_hull, width_ratio)

	# 8. Apply bitwise operation between convex hull result and the input image
	convex_hull = cv2.cvtColor(convex_hull, cv2.COLOR_GRAY2RGB)
	masked = cv2.bitwise_and(img, convex_hull)
	print("8. Segmentation result")
	show_image(debug, masked, width_ratio)

	return masked
