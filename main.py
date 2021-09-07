import cv2
import segmentation as seg
import align
import display_image as dis
import centering as center
import surface

# Read image
input_img = cv2.imread("cards/card_input/input_back_bad_1.jpeg", cv2.IMREAD_COLOR)
# read full image template
full_template_img = cv2.imread("cards/card_full/back_full_1.jpg", cv2.IMREAD_COLOR)
# read image template
template_img = cv2.imread("cards/card_template/back_template_1.png", cv2.IMREAD_COLOR)

# Apply centering grading
centering_grading_img = center.centering_grading(
        input_img=input_img,
        full_template=full_template_img,
        center_template=template_img,
        debug=False)

dis.show_image(debug_state=True, image=centering_grading_img, show_area=1080)

# Apply Surface grading
surface_grading_img = surface.surface_grading(
        input_img=input_img,
        full_template=full_template_img,
        debug=False)
# Show surface grading result
dis.show_image(debug_state=True, image=surface_grading_img, show_area=1080)
