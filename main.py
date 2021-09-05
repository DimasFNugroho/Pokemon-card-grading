import cv2
import segmentation as seg
import align

img = cv2.imread("cards/card_input/input_back_9.png", cv2.IMREAD_COLOR)
fg_img = seg.white_background_segmentation(img, debug=True, width_ratio=1080)
