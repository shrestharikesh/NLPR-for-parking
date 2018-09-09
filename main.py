import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess


def preprocess(img):
    cv2.imshow("Input", img)
    imgBlurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # canny = cv2.Canny(gray, 100, 200)
    cv2.imshow("Sobel", sobelx)
    cv2.waitKey(0)
    ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Threshold",threshold_img)
    # cv2.waitKey(0)
    return threshold_img

def cleanPlate(plate):
    print("CLEANING PLATE. . .")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # thresh= cv2.dilate(gray, kernel, iterations=1)
    # !!!!!!!!!!!!!!binarizing image
    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
    # im1, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # if contours:
    #     areas = [cv2.contourArea(c) for c in contours]
    #     max_index = np.argmax(areas)
    #
    #     max_cnt = contours[max_index]
    #     max_cntArea = areas[max_index]
    #     x, y, w, h = cv2.boundingRect(max_cnt)
    #
    #     if not ratioCheck(max_cntArea, w, h):
    #         return plate, None
    #
    #     cleaned_final = thresh[y:y + h, x:x + w]
    #     cv2.imshow("Function Test",cleaned_final)
    #     return cleaned_final, [x, y, w, h]
    #
    # else:
    return thresh

def extract_contours(threshold_img):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    cv2.imshow("Morphed", morph_img_threshold)
    cv2.waitKey(0)

    _, contours,_ = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_NONE)
    return contours

def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    aspect = 4.7272
    min = 15 * aspect * 15  # minimum area
    max = 125 * aspect * 125  # maximum area
    rmin = 1.5
    rmax = 2.5
    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    else:
        print(ratio)
    return True

def validateRotationAndRatio(rect):
    (_, _), (width, height), rect_angle = rect

    if (width > height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15:
        return False

    if height == 0 or width == 0:
        return False

    area = height * width
    if not ratioCheck(area, width, height):
        return False
    else:
        return True


def cleanAndRead(img, contours):
    # count=0
    for i, cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(min_rect)
        # box = np.int0(box)
        # cv2.drawContours(img,[box],0,(0,0,255),2)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        #!!!!!!!!!!!!!!!!!!!!!rectangle detectgariraxa
        if validateRotationAndRatio(min_rect):
            print("number plate ho but not accurate. machine learning launu parxa accurate result ko lagi")
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]
            # img = cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow("Detected Plate", plate_img)
            cv2.waitKey(0)
            # 	count+=1
            clean_plate = cleanPlate(plate_img)
           
            # if rect:
            #     x1, y1, w1, h1 = rect
            # x, y, w, h = x + x1, y + y1, w1, h1
            cv2.imshow("Cleaned Plate", clean_plate)
            cv2.waitKey(0)


# !!!!!!!!!!!!!ya chai plate lai text detection ko lagi patauna function thapna baki xa


            # plate_im = Image.fromarray(clean_plate)
            # text = tess.image_to_string(plate_im, lang='nep')
            # print("Detected Text : ", text)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Detected Plate", img)
            cv2.waitKey(0)
            return clean_plate, plate_img

# print "No. of final cont : " , count
def segment(image, RGBimg):
    # gray=cv2.cvtColor(plate,cv2.COLOR_BW2GRAY)
    _, contours,_ = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    idx =0 
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        roi=RGBimg[y:y+h,x:x+w]
        if(w>10 and h>10):
            cv2.imwrite(str(idx) + '.jpg', roi)

if __name__ == '__main__':
    print("DETECTING PLATE . . .")
    img = cv2.imread("img.jpg")
    threshold_img = preprocess(img)
    contours = extract_contours(threshold_img)
    plate,img = cleanAndRead(img, contours)
   
    segment(plate, img)
