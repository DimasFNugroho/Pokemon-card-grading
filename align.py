import cv2
import numpy as np
import imutils
import display_image as dis

def normalize_dimension(img1, img2):
	# calculate the difference of the image dimensions
	(h1, w1) = img1.shape[:2]
	(h2, w2) = img2.shape[:2]

	if h1 > h2:
		scale_percent = h2/h1 # percent of original size
		width = int(img1.shape[1] * scale_percent)
		height = int(img1.shape[0] * scale_percent)
		dim = (width, height)
		# resize image
		img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
	else:
		scale_percent = h1/h2
		width = int(img2.shape[1] * scale_percent)
		height = int(img2.shape[0] * scale_percent)
		dim = (width, height)
		# resize image
		img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
	return img1, img2

def align_image(img1, img2, maxFeatures=50000, keepPercent=2, debug=False):
	img1, img2 = normalize_dimension(img1, img2)
	# Initiate ORB detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with ORB
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
	matchedVis = imutils.resize(matchedVis, width=img1.shape[0])

	dis.show_image(debug, matchedVis)

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
	return aligned
