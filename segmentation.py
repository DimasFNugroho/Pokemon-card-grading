import cv2
import numpy as np
import imutils
import display_image as dis

def white_background_segmentation(img, debug=False, width_ratio=1080):

	# 1. convert the image to RGB
	if debug == True:
		print("1. Read_image")
	dis.show_image(debug, img, width_ratio)

	# 2.  Apply Median Blur
	median = cv2.medianBlur(img, 13)
	if debug == True:
		print("2. Apply median filter")
	dis.show_image(debug, median, width_ratio)

	# 3. get negative image
	img_neg = cv2.bitwise_not(median)
	if debug == True:
		print("3. Get inverted image")
	dis.show_image(debug, img_neg, width_ratio)

	# 4. Convert the image to gray-scale
	gray = cv2.cvtColor(img_neg, cv2.COLOR_BGR2GRAY)
	if debug == True:
		print("4. Get grayscale")
	dis.show_image(debug, gray, width_ratio)

	# 5. Apply thresholding
	(thresh, img_thresh) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
	h, w = img_thresh.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	if debug == True:
		print("flood")
	cv2.floodFill(img_thresh, mask, (0,0), 127)
	dis.show_image(debug, img_thresh, width_ratio)

	(thresh, img_thresh_white) = cv2.threshold(img_thresh, 250, 255, cv2.THRESH_BINARY)
	if debug == True:
		print("white")
	dis.show_image(debug, img_thresh_white, width_ratio)

	img_thresh_neg = cv2.bitwise_not(img_thresh)
	(thresh, img_thresh_black) = cv2.threshold(img_thresh_neg, 250, 255, cv2.THRESH_BINARY)
	if debug == True:
		print("black")
	dis.show_image(debug, img_thresh_black, width_ratio)

	img_thresh = img_thresh_white + img_thresh_black
	if debug == True:
		print("5. Apply thresholding")
	dis.show_image(debug, img_thresh, width_ratio)

	# 6. Find the edges
	edges = cv2.Canny(img_thresh, 100, 200)
	if debug == True:
		print("6. Get edges")
	dis.show_image(debug, edges, width_ratio)

	contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# 7. Find the convex hull from contours and draw it on the original image.
	convex_hull = edges
	for i in range(len(contours)):
		hull = cv2.convexHull(contours[i])
	cv2.drawContours(convex_hull, [hull], -1, (255, 0, 0), -1)
	if debug == True:
		print("7. Get convex hull")
	dis.show_image(debug, convex_hull, width_ratio)

	# 8. Apply bitwise operation between convex hull result and the input image
	convex_hull = cv2.cvtColor(convex_hull, cv2.COLOR_GRAY2RGB)
	masked = cv2.bitwise_and(img, convex_hull)
	if debug == True:
		print("8. Segmentation result")
	dis.show_image(debug, masked, width_ratio)

	return masked
