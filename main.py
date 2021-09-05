import cv2
import segmentation as seg
import align

# Read image
input_img = cv2.imread("cards/card_input/input_back_9.png", cv2.IMREAD_COLOR)
# Apply segmentation
fg_img = seg.white_background_segmentation(input_img, debug=False, width_ratio=1080)

# Apply image alignment with the image template
template_img = cv2.imread("cards/card_full/back_full_1.jpg", cv2.IMREAD_COLOR)
aligned_image = align.align_image(fg_img, template_img, maxFeatures=50000, keepPercent=2, debug=True)

seg.show_image(debug_state=True, image=aligned_image, show_area=1080)
