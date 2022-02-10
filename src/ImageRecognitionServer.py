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

# Main runtime
def main():
    sr = SymRec(WEIGHT_PATH)
    msg = sr.ProcessSourceImages(IMAGE_PATH)

main()
