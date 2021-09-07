import cv2
import numpy as np
import imutils

import display_image as dis
import align
import segmentation as seg
#
# Get contours and use convex hull to
# get the full region of the foreground pixels
#
def get_contour(image, debug=False):
# 1.  Apply Median Blur
	median = cv2.medianBlur(image, 5)
	dis.show_image(debug_state=debug, image=median, show_area=1080)

	# 3. Convert the image to gray-scale
	gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
	dis.show_image(debug_state=debug, image=gray, show_area=1080)

	# 4. Apply thresholding
	(thresh, img_thresh) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
	dis.show_image(debug_state=debug, image=img_thresh, show_area=1080)

	# 5. Find the contours
	contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# 6. Find the convex hull from contours and draw it on the original image.
	convex_hull = img_thresh
	for i in range(len(contours)):
		hull = cv2.convexHull(contours[i])
		cv2.drawContours(convex_hull, [hull], -1, (255, 0, 0), -1)
	dis.show_image(debug_state=debug, image=convex_hull, show_area=1080)
	return convex_hull

#
# Shift image to get full card appearance
#
def shift_image(aligned_image, debug=False):
	# Get the contour of the aligned image
	aligned_image_contour = get_contour(aligned_image, debug=debug)
	M = cv2.moments(aligned_image_contour)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

# compute the center of the contour
	convex_hull_rgb = cv2.cvtColor(aligned_image_contour, cv2.COLOR_GRAY2RGB)
# draw the center of the convex hull on the image
	cv2.circle(convex_hull_rgb, (cX, cY), 1, (100, 0, 100), -1)
	dis.show_image(debug_state=debug, image=convex_hull_rgb, show_area=1080)

	img_center_x = int(convex_hull_rgb.shape[1]/2)
	img_center_y = int(convex_hull_rgb.shape[0]/2)
# Draw the center of the image window
	cv2.circle(convex_hull_rgb, (img_center_x, img_center_y), 1, (0, 0, 0), -1)
	dis.show_image(debug_state=debug, image=convex_hull_rgb, show_area=1080)

	# Prepare the translation matrix
	x_shift = 0
	y_shift = 0
	if cX < img_center_x:
		x_shift = abs(cX - img_center_x) * 2
	if cX >= img_center_x:
		x_shift = abs(cX - img_center_x) * (-2)

	if cY < img_center_y:
		y_shift = abs(cY - img_center_y) * 2
	if cY >= img_center_y:
		y_shift = abs(cY - img_center_y) * (-2)

	translation_matrix = np.float32([ [1,0,x_shift], [0,1,y_shift] ])

# translate aligned image
	img_translation = cv2.warpAffine(
                aligned_image,
                translation_matrix,
                (img_center_x * 2, img_center_y * 2)
                )

	dis.show_image(debug_state=debug, image=img_translation, show_area=1080)
	return img_translation

def restore_pixels(debug, fg_img, translated_img):
	# Normalize image dimension
	# calculate the difference of the image dimensions
	(h_t, w_t) = translated_img.shape[:2]
	(h_o, w_o) = fg_img.shape[:2]

	width = 0
	height = 0

	if h_t > h_o:
		scale_percent = h_o/h_t # percent of original size
		width = int(translated_img.shape[1] * scale_percent)
		height = int(translated_img.shape[0] * scale_percent)
		dim = (width, height)
		# resize image
		translated_img = cv2.resize(translated_img, dim, interpolation = cv2.INTER_AREA)
	if h_o > h_t:
		scale_percent = h_t/h_o
		width = int(fg_img.shape[1] * scale_percent)
		height = int(fg_img.shape[0] * scale_percent)
		dim = (width, height)
		# resize image
		fg_img = cv2.resize(fg_img, dim, interpolation = cv2.INTER_AREA)
	dis.show_image(debug_state=debug, image=translated_img, show_area=1080)
	dis.show_image(debug_state=debug, image=fg_img, show_area=1080)

	final_aligned_image = align.align_image(img1=fg_img, img2=translated_img, debug=debug)
	return final_aligned_image

def get_center_template(blank_image):
	GREEN_MIN = np.array([36, 25, 25],np.uint8)
	GREEN_MAX = np.array([70, 255,255],np.uint8)

	hsv_img = cv2.cvtColor(blank_image,cv2.COLOR_BGR2HSV)
	frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)

	cnts = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	c=cnts[0]

	M = cv2.moments(c)
	cx = int((M["m10"] / M["m00"]))
	cy = int((M["m01"] / M["m00"]))

	return (cx, cy)

def centering_calculation(img1, img2, maxFeatures=1000, keepPercent=1, debug=False):
	# Initiate ORB detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(des1, des2, None)

	# sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]

	# Visualize
	matchedVis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
	matchedVis = imutils.resize(matchedVis, width=1000)

	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kp1[m.queryIdx].pt
		ptsB[i] = kp2[m.trainIdx].pt
	# compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

	# use the homography matrix to align the images
	(h, w) = img2.shape[:2]
	# align image
	aligned = cv2.warpPerspective(img1, H, (w, h))

	# mask with black image
	blank_image = np.zeros((img1.shape[0], img1.shape[1],3), np.uint8)
	cx = int(img1.shape[1]/2) # last center from template
	cy = int(img1.shape[0]/2) # last center from template

	cv2.circle(blank_image, (cx, cy), 10, (0,255,0), -1)

	blank_image_aligned = cv2.warpPerspective(blank_image, H, (w, h))
	tempalte_cx, template_cy = get_center_template(blank_image_aligned)
	cv2.circle(aligned, (tempalte_cx, template_cy), 1, (0,255,0), -1)

	original_cx, original_cy = int(aligned.shape[1]/2), int(aligned.shape[0]/2)

	a = np.array((tempalte_cx, template_cy))
	b = np.array((original_cx, original_cy))
	distance = dist = np.linalg.norm(a-b)

	(h4, w4) = aligned.shape[:2]
	cv2.circle(aligned,(int(w4/2),int(h4/2)), 1, (255,0,0), -1)
	#
	# Show the distance between the input image center and the template image center
	#
	print('euclidian distance: ', distance)

	return aligned

#
# Centering grading function
#
def centering_grading(input_img, full_template, center_template, debug=False):
	print("#")
	print("# Apply Centering Grading")
	print("#")
	print(" ")
	# Apply Segmentation
	fg_img = seg.white_background_segmentation(
                input_img,
                debug = debug,
                width_ratio=1080)

	# Apply image alignment from the input image to the template image
	aligned_image = align.align_image(
                fg_img,
                full_template,
                maxFeatures=50000,
                keepPercent=2,
                debug=debug)

	# Shift image to get the full appearance of the image
	shifted_image = shift_image(aligned_image, debug=debug)

# Restore the cropped image pixels
	restored_image = restore_pixels(
                debug=debug,
                fg_img=fg_img,
                translated_img=shifted_image)
	dis.show_image(debug_state=debug,
                image=restored_image,
                show_area=1080)
	# Apply Gaussian blur to the center_template image
	blur_center_template = cv2.GaussianBlur(center_template, (5,5), 0)
	# Apply centering calculation
	image_centering_result = centering_calculation(
                img1=blur_center_template,
                img2=input_img,
                maxFeatures=10000,
                keepPercent=5,
                debug=debug)

	dis.show_image(debug_state=debug,
                image=image_centering_result,
                show_area=1080)
	return image_centering_result
