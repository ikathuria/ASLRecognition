"""Visualizing the results with OpenCV."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("setting up, please wait...")

import cv2
import numpy as np
from keras.models import load_model
# from image_processing import run_avg, segment

# accumulated weight
accumWeight = 0.5

# path
latest_model = "model/" + "ASL_model2.h5"

# labels in order of training output
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5:'F', 6:'G', 7:'H',
          8: 'I'}
#  9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
#           15: 'P', 16: 'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W',
#           23:'X', 24: 'Y', 25: 'Z'}


def load_weights():
    """Load Model Weights.
    
    Returns:
        the loaded model if available, otherwise None.
    """
    try:
        model = load_model(latest_model)
        return model

    except Exception as e:
        return None


def getPredictedClass(model):
    """Get the predicted class.
    
    Args:
        model: the loaded model.
    
    Returns:
        the predicted class.
    """
    image = cv2.imread("Temp.png")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 100))
    gray_image = gray_image.reshape(1, 100, 100, 1)

    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)
    
    print(predicted_class)

    return labels[predicted_class].upper()


cap = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
top, right, bottom, left = 10, 310, 310, 610

num_frames = 0

model = load_weights()

while True:
    ret, frame = cap.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    cv2.imwrite('Temp.png', roi)

    predictedClass = getPredictedClass(model)

    cv2.putText(clone, str(predictedClass), (70, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", clone)

    num_frames += 1

    # on keypress
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break
    elif keypress == ord("c"):
        num_frames = 0

cap.release()
cv2.destroyAllWindows()
