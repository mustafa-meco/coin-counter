import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture("https://192.168.1.3:4343/video")
cap.set(3, 640)
cap.set(4, 480)



def empty(a):
    pass

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshold1", "Settings", 50, 255, empty)
cv2.createTrackbar("Threshold2", "Settings", 100, 255, empty)
cv2.createTrackbar("minArea", "Settings", 150, 1500, empty)


def preProcessing(img):
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgBlur = cv2.GaussianBlur(imgGray,(5,5),3)
    imgBlur = cv2.GaussianBlur(img,(5,5),3)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Settings")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Settings")
    imgCanny = cv2.Canny(imgBlur, threshold1, threshold2)
    imgPre = imgCanny
    kernel = np.ones((3,3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel) 
    return imgPre


while True:

    myColorFinder = ColorFinder(False)

    totalMoney = 0
    Pounds = 0
    Halfs = 0
    quarters = 0

    success, img = cap.read()
    # cut the top of the image
    img = img[10:480, 0:640]
    imgPre = preProcessing(img)
    threshold3 = cv2.getTrackbarPos("minArea", "Settings")

    imgContours, conFound = cvzone.findContours(img, imgPre,minArea=threshold3)

    if conFound:
        for contour in conFound:
            peri = cv2.arcLength(contour["cnt"], True)
            approx = cv2.approxPolyDP(contour["cnt"], 0.02*peri, True)
            # print(len(approx))
            if len(approx) > 5:
                area = contour["area"]
                
                if area <= 550:
                    totalMoney += 0.25
                    quarters += 1
                elif area <= 690:
                    totalMoney += 0.5
                    Halfs += 1
                elif area > 690:
                    totalMoney += 1
                    Pounds += 1

    # print("totalMoney:", totalMoney)
    # print("Pounds:", Pounds)
    # print("Halfs:", Halfs)

    imgStacked = cvzone.stackImages([img, imgPre, imgContours], 2, 1)
    cvzone.putTextRect(imgStacked, "Total Money: " + str(totalMoney) + " L.E.", (650, 550), colorR=(0, 200, 0))
    cvzone.putTextRect(imgStacked, "1 Pound: " + str(Pounds) + " coins", (650, 600), colorR=(0, 200, 0))
    cvzone.putTextRect(imgStacked, "0.5 Pound: " + str(Halfs) + " coins", (650, 650), colorR=(0, 200, 0))
    cvzone.putTextRect(imgStacked, "0.25 Pound: " + str(quarters) + " coins", (650, 700), colorR=(0, 200, 0))

    cv2.imshow("ImageStacked", imgStacked)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break