from my_module import yolo_detection as myf
import cv2


if __name__ == '__main__':
    image = cv2.imread("coke.png")
    myf.get_cordinate(image)