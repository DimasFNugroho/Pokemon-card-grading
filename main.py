import cv2
import segmentation as seg

img = cv2.imread("cards/card_input/input_back_1.png", cv2.IMREAD_COLOR)
fg_img = seg.white_background_segmentation(img, debug=True, show_ratio=0.8)

img = cv2.imread("cards/card_input/input_back_9.png", cv2.IMREAD_COLOR)
fg_img = seg.white_background_segmentation(img, debug=True, show_ratio=0.8)

img = cv2.imread("cards/card_input/input_back_bad_1.jpeg", cv2.IMREAD_COLOR)
fg_img = seg.white_background_segmentation(img, debug=True, show_ratio=0.8)

img = cv2.imread("cards/card_input/input_back_bad_2.jpeg", cv2.IMREAD_COLOR)
fg_img = seg.white_background_segmentation(img, debug=True, show_ratio=0.2)

img = cv2.imread("cards/card_input/input_cleffa.png", cv2.IMREAD_COLOR)
fg_img = seg.white_background_segmentation(img, debug=True, show_ratio=0.8)

img = cv2.imread("cards/card_input/input_lurantis.png", cv2.IMREAD_COLOR)
fg_img = seg.white_background_segmentation(img, debug=True, show_ratio=0.8)
