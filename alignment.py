import cv2
import numpy as np
from matplotlib import pyplot as plt
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

  if debug == True:
    # Show result
    plt.figure(figsize=(15, 15))
    plt.imshow(matchedVis)
    plt.title('matchedVis')

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

  # 
# Preprocess the image and apply image segmentation
# to get the foreground pixels.
#
# You may try the badly trimmed input image or the better one
#
 
# Read image 
img = cv2.imread("cards/card_input/inpud_back_bad_1.jpeg", cv2.IMREAD_COLOR)
# img = cv2.imread('Back_input.png', cv2.IMREAD_COLOR)
# img = cv2.imread('Back_input7.png', cv2.IMREAD_COLOR)
original = img
 
debug = True
 
# 1. convert the image to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
if debug == True:
  # Show RGB image
  plt.figure(figsize=(15, 15))
  plt.imshow(img, cmap='gray')
  plt.title('RGB')
 
# 2.  Apply Median Blur
median = cv2.medianBlur(img, 13)
if debug == True:
  # Show Median Blur result
  plt.figure(figsize=(15, 15))
  plt.imshow(median)
  plt.title('median')
 
# 3. get negative image
img_neg = cv2.bitwise_not(median)
if debug == True:
  # Show negative image
  plt.figure(figsize=(15, 15))
  plt.imshow(img_neg, cmap='gray')
  plt.title('negative')
 
# 4. Convert the image to gray-scale
gray = cv2.cvtColor(img_neg, cv2.COLOR_BGR2GRAY)
if debug == True:
  # Show grayscale
  plt.figure(figsize=(15, 15))
  plt.imshow(gray, cmap='gray')
  plt.title('grayscale')
 
# 5. Apply thresholding
(thresh, img_thresh) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
if debug == True:
  # Show thresholding result
  plt.figure(figsize=(15, 15))
  plt.imshow(img_thresh, cmap='gray')
  plt.title('threshold')
 
# 6. Find the contours
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# 7. Find the convex hull from contours and draw it on the original image.
convex_hull = img_thresh
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    cv2.drawContours(convex_hull, [hull], -1, (255, 0, 0), -1)
if debug == True:
  # Show convex hull result
  plt.figure(figsize=(15, 15))
  plt.imshow(convex_hull, cmap='gray')
  plt.title('contours')
 
# 8. Apply bitwise operation between convex hull result and the input image
convex_hull = cv2.cvtColor(convex_hull, cv2.COLOR_GRAY2RGB)
masked = cv2.bitwise_and(img, convex_hull)
if debug == True:
  # Show final segmentation result
  plt.figure(figsize=(15, 15))
  plt.imshow(masked)
  plt.title('edges')

# 
# Warp the forground to the image template
#

debug = True
img1 = masked

img2 = cv2.imread('back_full_1.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

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

# 1. Align image
aligned_image = align_image(img1, img2, debug=True)
if debug == True:
  # Show result
  plt.figure(figsize=(15, 15))
  plt.imshow(aligned_image, cmap='gray')
  plt.title('aligned_image')

#
# Get contours and use convex hull to
# get the full region of the foreground pixels
#

debug = True
# 1. convert the image to RGB
img = aligned_image
if debug == True:
  # Show RGB image
  plt.figure(figsize=(15, 15))
  plt.imshow(img, cmap='gray')
  plt.title('RGB')

# 2.  Apply Median Blur
median = cv2.medianBlur(img, 5)
if debug == True:
  # Show Median Blur result
  plt.figure(figsize=(15, 15))
  plt.imshow(median)
  plt.title('median')

# 3. Convert the image to gray-scale
gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
if debug == True:
  # Show grayscale
  plt.figure(figsize=(15, 15))
  plt.imshow(gray, cmap='gray')
  plt.title('grayscale')

# 4. Apply thresholding
(thresh, img_thresh) = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
if debug == True:
  # Show thresholding result
  plt.figure(figsize=(15, 15))
  plt.imshow(img_thresh, cmap='gray')
  plt.title('threshold')

# 5. Find the contours
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 6. Find the convex hull from contours and draw it on the original image.
convex_hull = img_thresh
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    cv2.drawContours(convex_hull, [hull], -1, (255, 0, 0), -1)
if debug == True:
  # Show convex hull result
  plt.figure(figsize=(15, 15))
  plt.imshow(convex_hull, cmap='gray')
  plt.title('convex_hull')

# 
# The translation contains only the image with the cropped pixels.
# However, the position of the image is assumed to be correct.
# 
# Then, the warping between the previously processed foreground pixels
# and the translated image is applied.
#
# Finally, the cropped pixels is restored, as shown in the following.
#

debug=True
# Normalize image dimension
# calculate the difference of the image dimensions
(h_t, w_t) = img_translation.shape[:2]
(h_o, w_o) = masked.shape[:2]

width = 0
height = 0

if h_t > h_o:
  scale_percent = h_o/h_t # percent of original size
  width = int(img_translation.shape[1] * scale_percent)
  height = int(img_translation.shape[0] * scale_percent)
  dim = (width, height)
  # resize image
  img_translation = cv2.resize(img_translation, dim, interpolation = cv2.INTER_AREA)
if h_o > h_t:
  scale_percent = h_t/h_o
  width = int(masked.shape[1] * scale_percent)
  height = int(masked.shape[0] * scale_percent)
  dim = (width, height)
  # resize image
  masked = cv2.resize(masked, dim, interpolation = cv2.INTER_AREA)

if debug == True:
  # Show result
  plt.figure(figsize=(15, 15))
  plt.imshow(img_translation, cmap='gray')
  plt.title('translated image')

# original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
# 1. Align image
final_aligned_image = align_image(masked, img_translation, debug=True)
if debug == True:
  # Show result
  plt.figure(figsize=(15, 15))
  plt.imshow(final_aligned_image, cmap='gray')
  plt.title('final alignment result')

if debug == True:
  # Show result
  plt.figure(figsize=(15, 15))
  plt.subplot(1, 2, 1), plt.imshow(final_aligned_image)
  plt.subplot(1, 2, 2), plt.imshow(img2)
  plt.show()
