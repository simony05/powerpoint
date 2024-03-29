import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import pyautogui

def get_args():
    screenWidth, screenHeight = pyautogui.size()
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=screenWidth)
    parser.add_argument("--height", help='cap height', type=int, default=screenHeight)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    # list to keep track of data points for identifying swiping motion
    datapoints = []

    # counter to make sure clicking motion doesn't double click by making clicks every 2 frames
    counter = 0

    screenWidth, screenHeight = pyautogui.size()

    while True:
        fps = cvFpsCalc.get()
        #print(fps)

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                #print(hand_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                #print(pre_processed_landmark_list)

                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                
                pointer = keypoint_classifier_labels[hand_sign_id]
                moving = point_history_classifier_labels[most_common_fg_id[0][0]]

                # coordinate positions
                wrist1 = landmark_list[0]
                index1 = landmark_list[8]
                middle1 = landmark_list[12]

                # for clicking motion, index finger bends down so less pixels
                indexbase = landmark_list[5]

                if (pointer == 'Pointer'):
                    pyautogui.moveTo(index1[0], index1[1], duration = 0.1, tween = pyautogui.easeInOutQuad)
                    # index len and dist btwn index base and wrist are abt the same
                    # when the index is half the length it means finger is bent, clicking motion
                    if (counter == 4):
                        if (abs(index1[1] - indexbase[1]) < 0.5 * abs(indexbase[1] - wrist1[1])):
                            print('clicked')
                            counter = 0
                    else:
                        counter += 1

                if (pointer == 'Pointer' and moving == 'Move'):
                    datapoints.append([wrist1, index1, middle1])

                # swipe must be at least 4 frames long
                # wrist, index tip, and middle tip at 1st frame and 4th frame are within 50px
                # x-value distance traveled must be at least .15 of the screen
                if (len(datapoints) > 3 and abs(datapoints[0][0][0] - datapoints[3][0][0]) > (screenWidth * 0.15)):
                    swipe = True
                    left = True if (datapoints[0][0][0] > datapoints[3][0][0]) else False 
                    for i in range(2):
                        # gives leeway if the start isn't exactly inline
                        # as long as some part of the swipe is straight it will count
                        if (abs(datapoints[i][0][1] - datapoints[i + 2][0][1]) > (0.1 * screenHeight)
                            or abs(datapoints[i][1][1] - datapoints[i+ 2][1][1]) > (0.1 * screenHeight)
                            or abs(datapoints[i][2][1] - datapoints[i + 2][2][1]) > (0.1 * screenHeight)):
                            swipe = False
                            break
                    if swipe is True:
                        if left:
                            print('valid left swipe')
                        else:
                            print('valid right swipe')
                    datapoints.clear()
                else:
                    # swipe failed, restart
                    if len(datapoints) > 3:
                        datapoints.clear()
        else:
            # when no hand is in the frame
            datapoints.clear()
    
        # Popup window
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

if __name__ == '__main__':
    main()
