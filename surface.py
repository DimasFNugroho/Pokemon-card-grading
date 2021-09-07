import cv2
import numpy as np
import imutils

import display_image as dis
import align
import segmentation as seg

def clache(aligned_image, full_template, debug=False):
#
# First align both image to make sure the features of each image
# are aligned at the same position.
#
# Apply CLACHE for Histogram Equalization. So the appearance of the image are
# similar
#
	aligned_image_gray = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2GRAY)

	template_gray = cv2.cvtColor(full_template, cv2.COLOR_RGB2GRAY)

	clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(4,4))
	cl1 = clahe.apply(aligned_image_gray)
	cl2 = clahe.apply(template_gray)
	cl2 = cv2.blur(cl2,(5,5))

	dis.show_image(debug_state=debug, image=cl1, show_area=1080)
	dis.show_image(debug_state=debug, image=cl2, show_area=1080)

	return cl1, cl2

def surface_grading(input_img, full_template, debug=False):
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
	cl1, cl2 = clache(aligned_image, full_template, debug=debug)

	# Get image difference
	image_diff = cv2.absdiff(cl1, cl2)

	# Apply thresholding of the image difference
	(thresh, img_thresh) = cv2.threshold(image_diff,55, 255, cv2.THRESH_BINARY)

	# Errode image to get the rid of edges
	kernel = np.ones((5, 5), np.uint8)
	erroded_image = cv2.erode(img_thresh, kernel)
	dis.show_image(debug_state=debug, image=erroded_image, show_area=1080)


	# Calculate pixel difference
	hist = cv2.calcHist([erroded_image],[0],None,[256],[0,256])
	bad_surface_percentage = (hist[255]/hist[0])*100
	print(hist[0])
	print(hist[255])
	print("Bad Surface Percentage: ", bad_surface_percentage[0], "%")

	# then draw the bad surface part on the input image
	image_surface_grading_result = aligned_image.copy()

	contours,hierarchy = cv2.findContours(erroded_image, 1, 2)
	cv2.drawContours(image_surface_grading_result, contours, -1, (0,255,0), 2)
	cv2.drawContours(image_surface_grading_result, contours, -1, (255,0,0), -1)
	return image_surface_grading_result
