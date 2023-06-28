from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from recognizer import prediction, model


cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
resd = ''
while True:

        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 200, 255)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        displayCnt = None

        # выделить рамку вокруг цифр (в физмат корпусе номера кабинетов ввиде: белый фон -> черная рамка-> цифры
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                displayCnt = approx
                break

        # вырезать roi цифр и скормить сетке
        # roi определяем на edged а вырезаем из gray, т.к модель обучена на чб
        if displayCnt is not None:
            warped = four_point_transform(edged, displayCnt.reshape(4, 2))
            gray = four_point_transform(gray, displayCnt.reshape(4, 2))
            cnts, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
          #  раскоментить для просмотра roi
          #  cv2.imshow("warped", thresh1)

            # вычисляем bound box для цифр
            digits = []
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if (20 <= w < 150) and (50 < h < 100):
                    image = gray[y - 15:y + h + 3, x - 15:x + w]
                    if image.shape[0] > 28 and image.shape[1] > 28:
                        digits.append(c)


            if digits:
                tmp =''
                digits = contours.sort_contours(digits, method="left-to-right")[0]
                for c in digits:
                    # extract the digit ROI
                    (x, y, w, h) = cv2.boundingRect(c)
                    image = gray[y - 10:y + h+5, x - 15:x + w]
                    if image.shape[0] > 28 and image.shape[1] > 28:
                        res, prob = prediction(image, model)
                        tmp += str(res)
                        resd = tmp

        cv2.putText(img, f"Pred: {resd}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
        cv2.imshow("canny", img)
        cv2.waitKey(25)
