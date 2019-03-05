import pytesseract
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("numbers/55_r.png")
m, n = img.shape[0:2]

blur = cv2.GaussianBlur(img, (0, 0), 2)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

coutours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
print(coutours)

kerne1 = np.ones((3, 3), np.uint8)
img_erosin = cv2.erode(binary, kerne1, iterations=1)

cv2.imshow("ers", img_erosin)

out = img_erosin
s = pytesseract.image_to_string(out, lang="num")
print(s)
cv2.waitKey()
