import cv2
import numpy as np
import imutils

import display_image as dis
import align
import segmentation as seg
import surface

def edges_grading(input_img, full_template, debug=False):
	print("#")
	print("# Apply Surface Grading")
	print("#")
	print(" ")
	# Apply segmentation
	fg_img = seg.white_background_segmentation(
                img=input_img,
                debug=debug,
                width_ratio=1080)

	# Apply image alignment
	aligned_image = align.align_image(
                img1 = fg_img,
                img2 = full_template,
                maxFeatures=1000,
                keepPercent=2,
                debug=debug)

	# Apply CLACHE
	cl1, cl2 = surface.clache(aligned_image, full_template, debug=debug)
	# Apply blur to template
	cl2 = cv2.blur(cl2,(5,5))
	# Get image difference
	image_diff = cv2.absdiff(cl1, cl2)
	# extract aligned card contour
	aligned_img_gray = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2GRAY)
	(thresh, full_card_thresh) = cv2.threshold(aligned_img_gray, 10, 255, cv2.THRESH_BINARY)
	(thresh, aligned_bad_surface) = cv2.threshold(image_diff, 100, 255, cv2.THRESH_BINARY)

	dis.show_image(debug_state=debug, image=image_diff, show_area=1080)
	dis.show_image(debug_state=debug, image=aligned_bad_surface, show_area=1080)
	# Find the contours
	contours, hierarchy = cv2.findContours(
                full_card_thresh,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
	# Find the convex hull from contours and draw it on the original image.
	convex_hull = full_card_thresh
	for i in range(len(contours)):
	    hull = cv2.convexHull(contours[i])
	    cv2.drawContours(convex_hull, [hull], -1, (255, 0, 0), -1)

	# create blank image with single-pixel-white-borders
	(h, w) = convex_hull.shape[:2]
	blank_image = convex_hull
	cv2.rectangle(blank_image,(0,0),(w,h),(0,0,0),3)

	# apply canny edge
	edges = cv2.Canny(convex_hull,100,200, 5)
	kernel = np.ones((5, 5), np.uint8)
	dilated_card_edges = cv2.dilate(edges,kernel,iterations = 1)

	#
	# Get bad edges result
	#
	bad_edges_result = cv2.bitwise_and(
                dilated_card_edges,
                dilated_card_edges,
                mask = aligned_bad_surface)

	kernel = np.ones((5, 5), np.uint8)
	dilated_bad_edges = cv2.dilate(edges, kernel, iterations = 1)

	#
	# Calculate histogram and then draw the bad surface part on the input image
	#
	fg_hist = cv2.calcHist([bad_edges_result],[0],None,[256],[0,256])
	bad_edges_pixels = fg_hist[255]

	true_line_hist = cv2.calcHist([dilated_card_edges],[0],None,[256],[0,256])
	edges_pixels = true_line_hist[255]

	bad_edges_percentage = (bad_edges_pixels / edges_pixels) * 100

	print(bad_edges_pixels)
	print(edges_pixels)
	print("Bad Edges Percentage: ", bad_edges_percentage[0], "%")

	edge_grading_result = aligned_image.copy()

	# Draw ideal card edges shape
	contours,hierarchy = cv2.findContours(dilated_card_edges, 1, 2)
	cv2.drawContours(edge_grading_result, contours, -1, (0,255,0), -1)

	# Draw bad edges of the card
	# Enhance the bad edges thickness
	bad_edges_result = cv2.dilate(bad_edges_result,kernel,iterations = 1)
	contours,hierarchy = cv2.findContours(bad_edges_result, 1, 2)
	cv2.drawContours(edge_grading_result, contours, -1, (255,0,0), -1)

	dis.show_image(debug_state=debug,
                image=edge_grading_result,
                show_area=1080)
	return edge_grading_result
