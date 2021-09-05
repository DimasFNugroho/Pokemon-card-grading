import cv2
import numpy as np
import imutils

def align_image(img1, img2, maxFeatures=50000, keepPercent=2, debug=False):
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
	matchedVis = imutils.resize(matchedVis, width=1000)

	show_image(debug, matchedVis, show_ratio=1)

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
