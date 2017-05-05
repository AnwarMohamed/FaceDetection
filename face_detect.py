#!/usr/bin/env python

import cv2
import sys
import argparse


def handle_single(parser, args):

    if args.image is None:
        parser.error("single mode requires --image")
    elif args.casc is None:
        parser.error("single mode requires --casc")

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(args.casc)

    # Read the image
    image = cv2.imread(args.image)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        grayImage,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)    
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)  

def handle_live(parser, args):
    pass

def handle_train(parser, args):
    pass    

if __name__ == '__main__':

    # Setting up argument parser
    parser = argparse.ArgumentParser("Face Detection using OpenCV")
    parser.add_argument("-m", "--mode", choices=['single','live', 'train'], required=True)

    # Arguments for single mode
    parser.add_argument("-i", "--image")
    parser.add_argument("-c", "--casc")

    args = parser.parse_args()

    if args.mode == 'single':
        handle_single(parser, args)

    elif args.mode == 'live':
        handle_live(parser, args)

    elif args.mode == 'train':
        handle_train(parser, args)
