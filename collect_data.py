"""Data Collection for gesture recognition model with OpenCV."""

import cv2
import numpy as np
import os
from image_processing import run_avg, segment

# accumulated weight
accumWeight = 0.5

# path
mode = "ASL"
directory = f"{mode}_data/"

# training labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
          'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
          'Y', 'Z']

# create the directories
if not os.path.exists(f"{mode}_data"):
    os.makedirs(f"{mode}_data")

for i in labels:
    if not os.path.exists(directory + i):
        os.makedirs(directory + i)

cap = cv2.VideoCapture(0)
num_frames = 0

while True:
    ret, frame = cap.read()

    # removing mirror image
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # getting count of existing images
    count = {
        "A": len(os.listdir(directory + "A")),
        "B": len(os.listdir(directory + "B")),
        "C": len(os.listdir(directory + "C")),
        "D": len(os.listdir(directory + "D")),
        "E": len(os.listdir(directory + "E")),
        "F": len(os.listdir(directory + "F")),
        "G": len(os.listdir(directory + "G")),
        "H": len(os.listdir(directory + "H")),
        "I": len(os.listdir(directory + "I")),
        "J": len(os.listdir(directory + "J")),
        "K": len(os.listdir(directory + "K")),
        "L": len(os.listdir(directory + "L")),
        "M": len(os.listdir(directory + "M")),
        "N": len(os.listdir(directory + "N")),
        "O": len(os.listdir(directory + "O")),
        "P": len(os.listdir(directory + "P")),
        "Q": len(os.listdir(directory + "Q")),
        "R": len(os.listdir(directory + "R")),
        "S": len(os.listdir(directory + "S")),
        "T": len(os.listdir(directory + "T")),
        "U": len(os.listdir(directory + "U")),
        "V": len(os.listdir(directory + "V")),
        "W": len(os.listdir(directory + "W")),
        "X": len(os.listdir(directory + "X")),
        "Y": len(os.listdir(directory + "Y")),
        "Z": len(os.listdir(directory + "Z")),
    }

    # printing the count in each set to the screen
    cv2.putText(clone, "A : " + str(count["A"]), (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "B : " + str(count["B"]), (10, 35),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "C : " + str(count["C"]), (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "D : " + str(count["D"]), (10, 65),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "E : " + str(count["E"]), (10, 80),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "F : " + str(count["F"]), (10, 95),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "G : " + str(count["G"]), (10, 110),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "H : " + str(count["H"]), (10, 125),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "I : " + str(count["I"]), (10, 140),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "J : " + str(count["J"]), (10, 155),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "K : " + str(count["K"]), (10, 170),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "L : " + str(count["L"]), (10, 185),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "M : " + str(count["M"]), (10, 200),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "N : " + str(count["N"]), (10, 215),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "O : " + str(count["O"]), (10, 230),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "P : " + str(count["P"]), (10, 245),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "Q : " + str(count["Q"]), (10, 260),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "R : " + str(count["R"]), (10, 275),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "S : " + str(count["S"]), (10, 290),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "T : " + str(count["T"]), (10, 305),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "U : " + str(count["U"]), (10, 320),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "V : " + str(count["V"]), (10, 335),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "W : " + str(count["W"]), (10, 350),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "X : " + str(count["X"]), (10, 365),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "Y : " + str(count["Y"]), (10, 380),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv2.putText(clone, "Z : " + str(count["Z"]), (10, 395),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    # coordinates of the Region Of Interest (ROI)
    top, right, bottom, left = 10, 310, 310, 610

    # drawing the ROI
    roi = frame[top:bottom, right:left]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 30:
        run_avg(gray, accumWeight)
        if num_frames == 1:
            print("\n[STATUS] please wait! calibrating...")
        elif num_frames == 29:
            print("[STATUS] calibration successfull...")
    else:
        # segment the hand region
        hand = segment(gray)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, segmented) = hand

            # draw the segmented region and display the frame
            cv2.drawContours(
                clone, [segmented + (right, top)], -1, (0, 0, 255))

            cv2.imshow("Threshold Image", thresholded)

    # bounding box
    cv2.rectangle(clone, (left, top), (right, bottom), (255, 0, 0), 1)

    # data collection frame
    cv2.imshow("Data Collection", clone)

    num_frames += 1

    # on keypress
    keypress = cv2.waitKey(10) & 0xFF
    if keypress == 27:  # esc
        break

    # recalibrate
    if keypress == ord("1"):
        num_frames = 0

    # take pictures
    if keypress == ord("a"):
        cv2.imwrite(directory + "A/" + str(count["A"]) + ".jpg", thresholded)
    if keypress == ord("b"):
        cv2.imwrite(directory + "B/" + str(count["B"]) + ".jpg", thresholded)
    if keypress == ord("c"):
        cv2.imwrite(directory + "C/" + str(count["C"]) + ".jpg", thresholded)
    if keypress == ord("d"):
        cv2.imwrite(directory + "D/" + str(count["D"]) + ".jpg", thresholded)
    if keypress == ord("e"):
        cv2.imwrite(directory + "E/" + str(count["E"]) + ".jpg", thresholded)
    if keypress == ord("f"):
        cv2.imwrite(directory + "F/" + str(count["F"]) + ".jpg", thresholded)
    if keypress == ord("g"):
        cv2.imwrite(directory + "G/" + str(count["G"]) + ".jpg", thresholded)
    if keypress == ord("h"):
        cv2.imwrite(directory + "H/" + str(count["H"]) + ".jpg", thresholded)
    if keypress == ord("i"):
        cv2.imwrite(directory + "I/" + str(count["I"]) + ".jpg", thresholded)
    if keypress == ord("j"):
        cv2.imwrite(directory + "J/" + str(count["J"]) + ".jpg", thresholded)
    if keypress == ord("k"):
        cv2.imwrite(directory + "K/" + str(count["K"]) + ".jpg", thresholded)
    if keypress == ord("l"):
        cv2.imwrite(directory + "L/" + str(count["L"]) + ".jpg", thresholded)
    if keypress == ord("m"):
        cv2.imwrite(directory + "M/" + str(count["M"]) + ".jpg", thresholded)
    if keypress == ord("n"):
        cv2.imwrite(directory + "N/" + str(count["N"]) + ".jpg", thresholded)
    if keypress == ord("o"):
        cv2.imwrite(directory + "O/" + str(count["O"]) + ".jpg", thresholded)
    if keypress == ord("p"):
        cv2.imwrite(directory + "P/" + str(count["P"]) + ".jpg", thresholded)
    if keypress == ord("q"):
        cv2.imwrite(directory + "Q/" + str(count["Q"]) + ".jpg", thresholded)
    if keypress == ord("r"):
        cv2.imwrite(directory + "R/" + str(count["R"]) + ".jpg", thresholded)
    if keypress == ord("s"):
        cv2.imwrite(directory + "S/" + str(count["S"]) + ".jpg", thresholded)
    if keypress == ord("t"):
        cv2.imwrite(directory + "T/" + str(count["T"]) + ".jpg", thresholded)
    if keypress == ord("u"):
        cv2.imwrite(directory + "U/" + str(count["U"]) + ".jpg", thresholded)
    if keypress == ord("v"):
        cv2.imwrite(directory + "V/" + str(count["V"]) + ".jpg", thresholded)
    if keypress == ord("w"):
        cv2.imwrite(directory + "W/" + str(count["W"]) + ".jpg", thresholded)
    if keypress == ord("x"):
        cv2.imwrite(directory + "X/" + str(count["X"]) + ".jpg", thresholded)
    if keypress == ord("y"):
        cv2.imwrite(directory + "Y/" + str(count["Y"]) + ".jpg", thresholded)
    if keypress == ord("z"):
        cv2.imwrite(directory + "Z/" + str(count["Z"]) + ".jpg", thresholded)

cap.release()
cv2.destroyAllWindows()
