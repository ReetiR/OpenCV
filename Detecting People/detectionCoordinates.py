import cv2
import sys
import datetime
import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from array import array
def distance_to_camera(known, focalLength, Width): #to calculate distance of object from camera
    return (focalLength*known) / Width

def findMarker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 20, 200)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if(cnts): #if marker found
        c = max(cnts, key = cv2.contourArea)
        print "Marker Coordinates"
        print cv2.minAreaRect(c)
        return cv2.minAreaRect(c)
    else: #no marker found
        return False

Dist = input("Enter marker distance(in inches):")
Width = input("Enter marker width(in inches):")

KNOWN_DISTANCE = Dist
KNOWN_WIDTH = Width


cascPath = 'haarcascade_upperbody.xml'
bodyCascade = cv2.CascadeClassifier(cascPath)

camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

(grabbed, frame) = camera.read()
marker = findMarker(frame)
if(marker!=False): #calculate midpoint of marker bounding box and focal length
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    markerX = (marker[0][0]+marker[1][0])/2
    ratio = KNOWN_WIDTH/(marker[1][0]-marker[0][0]) #known width vs pixel width
else:
    print "No suitable marker found. Cannot decide depth of image"


fig=plt.figure()
f, axScatter = plt.subplots()
axScatter.grid()

while True:
    text = "Empty Room"
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dist =0 

    body = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    if(marker!=False):
        xvals = array('f')
        yvals = array('f')
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if(marker!=False):
            #calculate x with respect to marker and y with respect to camera
            dist = distance_to_camera(KNOWN_WIDTH, focalLength, (x+w))
            x1= (((2*x+w)/2)-markerX) * ratio #multiple with the ratio of known vs pixel to get the real value
            y1= Dist-math.sqrt(abs(dist**2-x1**2)) #since distance of object to camera is straight line distance. Need pythagoras to get value along y axis side=sqrt(hypotenuse^2-base^2)
            print "Coordinates"
            print x1
            print y1
            xvals.append(x1)
            yvals.append(y1)
        text = "Person Detected"
    if(marker!=False):
    	r= random.uniform(0,1)
    	g= random.uniform(0,1)
    	b= random.uniform(0,1)
    	col=(r,g,b)
    	rgb = matplotlib.colors.colorConverter.to_rgb(col) #differentiate every iteration by color
        xnp= np.array(xvals)
        ynp= np.array(yvals)
        axScatter.scatter(xnp, ynp, s=100, color=rgb)
        ax = plt.axes()
        left,right = ax.get_xlim()
        low,high = ax.get_ylim()
        ax.arrow( left, 0, right -left, 0, length_includes_head = True, head_width = 0.15 )
        ax.arrow( 0, low, 0, high-low, length_includes_head = True, head_width = 0.15 ) 

    #display on frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(frame, "%.2f" % (dist),
        (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 0), 3)
    cv2.imshow('Video', frame)
    if(marker!=False):
        plt.draw()
        plt.pause(0.0001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
