from time import time

import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

####################################
classID =1   # 0 is fake and 1 is real
outputFolderPath = 'C:\liveliness detection\Dataset\DataCollect'
confidence = 0.8
save = True  #Flag to enable/disable saving of images and labels
blurThreshold = 30

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6 #Precision for normalized values.
####################################


cap = cv2.VideoCapture(0) #opens camera
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()
while True:
    success, img = cap.read()  #Captures a frame from the camera.
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False) #Detects faces in the image without drawing on it.

    #lists to store blur detection results and information for labels.
    listBlur = []  # True=clear False=blurry
    listInfo = []
    if bboxs: #bboxes contains a list of bounding boxes

        for bbox in bboxs:
            x, y, w, h = bbox["bbox"] #extracts the position and size of the bounding box for each face.
            score = bbox["score"][0] # extracts the confidence score of the detected face.


            # ------  Check the score --------
            if score > confidence:

                # ------  Adding an offset to the face Detected --------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3)

                # ------  To avoid values below 0 --------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ------  Find Blurriness --------
                imgFace = img[y:y + h, x:x + w] #extracts detected face
                cv2.imshow("Face", imgFace) #Displays an image in a window.
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())  #function that calculates the Laplacian of
                # the image. The Laplacian is a measure of edge sharpness. In general, a blurry image will have a lower
                # Laplacian value compared to a sharp image.
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ------  Normalize Values  --------
                ih, iw,_ = img.shape #img.shape is a tuple containing three elements: height, width, and number of channels
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint) #moving mid pt
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # ------  To avoid values above 1 --------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ------  Drawing boundiing box--------
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                   scale=2, thickness=3)
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                       scale=2, thickness=3)

        # ------  To Save --------
        if save:
            if all(listBlur) and listBlur != []:
                # ------  Save Image  --------
                timeNow = time()
                timeNow = str(timeNow).split('.') #converts the timestamp to a string and then splits it at the decimal
                # point (.) to separate the seconds and microseconds.
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img) #cv2.imwrite is an OpenCV function for
                # writing images to disk.
                # ------  Save Label Text File  --------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)

