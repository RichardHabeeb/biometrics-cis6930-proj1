################################################################################
# CIS 6930- Biometrics Project 1
# Richard Habeeb & Palak Dave
#
################################################################################

######################################################################
# IMPORTS
######################################################################
import cv2
import argparse
import math
import numpy as np

######################################################################
# CLASSES
######################################################################

######################################################################
# GLOBALS
######################################################################

######################################################################
# FUNCTIONS
######################################################################

#---------------------------------------
# main
#
#---------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classifier', help='Path to face detection cascade classifier xml.', default='../haarcascades/haarcascade_frontalface_default.xml')
    parser.add_argument('-d', '--distance', help='Scale eye-to-eye distance to this setting in pixels.', default=180)
    parser.add_argument('image', help='Image file to analyze.')
    args = parser.parse_args()

    # Locate all the faces in the image
    for (roi_gray, roi_color) in find_all_faces(args.image, args.classifier):

        # Extract the features from this roi
        features = find_face_features(roi_gray, roi_color)

        # Mark some important features
        cv2.rectangle(roi_color,
            (features['left_eye'][0], features['left_eye'][1]),
            (features['left_eye'][0] + 2, features['left_eye'][1] + 2),
            (0, 255, 0), 2)

        cv2.rectangle(roi_color,
            (features['right_eye'][0], features['right_eye'][1]),
            (features['right_eye'][0] + 2, features['right_eye'][1] + 2),
            (0, 255, 0), 2)

        cv2.rectangle(roi_color,
            (features['nose_tip'][0], features['nose_tip'][1]),
            (features['nose_tip'][0] + 2, features['nose_tip'][1] + 2),
            (0, 255, 0), 2)

        # Rotate the image to align the eyes. Recompute the new features
        rotated_color, new_features = align_eyes(features, roi_color, args.distance)

        # Mask off everything but the face ellipse
        masked_roi = mask_face(new_features, rotated_color)

        display_image(masked_roi)


#---------------------------------------
# find_all_faces
#
# Returns a list of rois.
#---------------------------------------
def find_all_faces(path, classifier):
    face_cascade = cv2.CascadeClassifier(classifier)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    rois = []
    for (x,y,w,h) in faces:
        rois.append((gray[y:y+h, x:x+w], img[y:y+h, x:x+w]))
    return rois

#---------------------------------------
# find_face_features
#
# Some machine learning algorithm that extracts face features given a roi
#---------------------------------------
def find_face_features(roi_gray, roi_color):

    # TEMPORARY IMPLEMENTATION! REWRITE!!!

    width = len(roi_gray)
    height = len(roi_gray[0])
    return {
        "left_eye":(width/4, height/4),
        "right_eye":(width*3/4, height/4 + 40),
        "nose_tip":(width/2, height*3/4)
    }

#---------------------------------------
# align_eyes
#
# Rotate and scale the image to that the eyes are level and that they are 'eye_distance_target' pixels apart.
#---------------------------------------
def align_eyes(features, roi, eye_distance_target):
    x1 = features['left_eye'][0]
    y1 = features['left_eye'][1]
    x2 = features['right_eye'][0]
    y2 = features['right_eye'][1]
    width = len(roi)
    height = len(roi[0])
    eye_distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    correction_angle = math.acos((x2-x1) / eye_distance)
    scale = eye_distance_target/eye_distance

    M = cv2.getRotationMatrix2D((width/2, height/2), 180*correction_angle/math.pi, scale)

    res = cv2.transform(np.array([[
        [features['left_eye'][0], features['left_eye'][1]],
        [features['right_eye'][0], features['right_eye'][1]],
        [features['nose_tip'][0], features['nose_tip'][1]],
        ]]), M)

    new_features = {
        "left_eye": (res[0][0][0], res[0][0][1]),
        "right_eye": (res[0][1][0], res[0][1][1]),
        "nose_tip": (res[0][2][0], res[0][2][1]),
    }
    return cv2.warpAffine(roi, M, (width,height)), new_features

#---------------------------------------
# mask_face
#
# Applies a ellipse mask to an roi given some face features
#---------------------------------------
def mask_face(features, roi):
    width = len(roi)
    height = len(roi[0])
    mask = np.zeros((height,width), np.uint8)
    cv2.ellipse(
        mask,
        ((features['right_eye'][0] + features['left_eye'][0])/2, (features['nose_tip'][1] + features['left_eye'][1])/2),
        ((features['right_eye'][0] - features['left_eye'][0])*3/4, (features['nose_tip'][1] - features['left_eye'][1])), #magic numbers
        0, 0, 360, (255,255,255), -1)
    return cv2.bitwise_and(roi, roi, mask=mask)

#---------------------------------------
# display_image
#
#---------------------------------------
def display_image(img, title='image', blocking=True):
    cv2.imshow(title, img)
    if(blocking):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#---------------------------------------
# display_side_by_side
#
#---------------------------------------
def display_side_by_side(img1, img2, *args, **kwargs):
    display_image(np.hstack((img1, img2)), *args, **kwargs)


######################################################################
# EXECUTION
######################################################################
if __name__ == "__main__":
    main()
