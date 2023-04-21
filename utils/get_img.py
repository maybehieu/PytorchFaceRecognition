import cv2 as cv
import datetime
import os
import sys

LIMIT = 20

print('WARNING: this tool doesn\'t implement face detecting module, please ensure your face is visible within camera frame!')
user_name = input('Enter username: ')
path = input('Enter folder path: ')
cam = cv.VideoCapture(0)
cnt = 0
while True:
    ret, frame = cam.read()
    if not ret: continue
    filename = user_name + '_' + str(cnt) + '.png'
    filepath = os.path.normpath(path + '/' + filename)
    print(path)
    cv.imshow('frame', frame)
    cv.waitKey(0)
    cv.imwrite(filepath, frame)
    cnt += 1
    filepath, filename = '', ''
    if cnt > LIMIT: break
sys.exit()