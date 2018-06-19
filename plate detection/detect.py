import cv2
import os
import numpy as np
import Preprocess
def main():
    img = cv2.imread('test.jpg')    #open images

    if img is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return 

    height, width, numChannels = img.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(img)         # preprocess to get grayscale and threshold images
    


    imgBlurred = cv2.GaussianBlur(img, (5,5), 0)		
    imgGray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
    imgSobelx = cv2.Sobel(imgGray,cv2.CV_8U,1,0,ksize=3)
    cv2.imshow("Sobel",imgSobelx)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()