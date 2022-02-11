# ImageRecognitionServer - Server that receives and stores images from RPi
# @author Lim Rui An, Ryan
# @version 1.0
# @since 2022-02-10
# @modified 2022-02-10

from SymbolRecognizer import SymbolRecognizer as SymRec

# Constant PATH variables
WEIGHT_PATH = "../weights/e40b16v8best.pt"
YOLO_PATH = "../yolov5"
IMAGE_PATH = "../img/1_5.jpg"

# Other Constants
ANNOTATION_CLASSES = ['1_blue', '2_green', '3_red', '4_white', '5_yellow', '6_blue', '7_green', '8_red', '9_white', 'a_red', 'b_green', 'bullseye', 'c_white', 'circle_yellow', 'd_blue', 'down_arrow_red', 'e_yellow', 'f_red', 'g_green', 'h_white', 'left_arrow_blue', 'right_arrow_green', 's_blue', 't_yellow', 'u_red', 'up_arrow_white', 'v_green', 'w_white', 'x_blue', 'y_yellow', 'z_red']
NUM_CLASSES = 31

# Main runtime
def main():
    # Get result from processed image
    sr = SymRec(WEIGHT_PATH, ANNOTATION_CLASSES, NUM_CLASSES)
    msg = sr.ProcessSourceImages(IMAGE_PATH)
    print("Detected: " + msg)
main()
