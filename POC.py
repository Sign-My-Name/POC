import mediapipe as mp
import cv2
import math
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

landmarks = ["WRIST", "THUMB_TIP", "INDEX_FINGER_TIP", "PINKY_TIP", "RING_FINGER_DIP"]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


het_distances = [0.188, 0.344, 0.304, 0.182, 0.008]
alef_distances = [0.195, 0.123, 0.121, 0.128, 0.133]

threshold = 0.10


def dist_cal(landmarks, hand_landmarks):
    distances = []
    x_wrist = hand_landmarks[mp_hands.HandLandmark.WRIST].x
    y_wrist = hand_landmarks[mp_hands.HandLandmark.WRIST].y
    z_wrist = hand_landmarks[mp_hands.HandLandmark.WRIST].z
    x_middle_base = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    y_middle_base = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    z_middle_base = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
    x_thumb = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP].x
    y_thumb = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    z_thumb = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP].z
    x_ring = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x
    y_ring = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y
    z_ring = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].z

    for landmark in landmarks[1:]:
        index = mp_hands.HandLandmark[landmark]
        x, y, z = hand_landmarks[index].x, hand_landmarks[index].y, hand_landmarks[index].z
        dist = distance(x_wrist, y_wrist, z_wrist, x, y, z)
        distances.append(dist)

    dist = distance(x_thumb, y_thumb, z_thumb, x_ring, y_ring, z_ring)
    distances.append(dist)

    return np.array(distances), distance(x_thumb, y_thumb, z_thumb, x_middle_base, y_middle_base, z_middle_base)


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                distances, thumb_to_middle = dist_cal(landmarks, hand_landmarks.landmark)

                het_diff, alef_diff = np.abs(distances - het_distances), np.abs(distances - alef_distances)

                if np.all(het_diff < threshold) and thumb_to_middle < 0.095:
                    label = "Het"
                elif np.all(alef_diff < threshold) and thumb_to_middle > 0.105:
                    label = "Alef"
                else:
                    label = "None"

                min_y = min(hand_landmarks.landmark, key=lambda lm: lm.y)
                cv2.putText(image, label, (
                    int(min_y.x * image.shape[1]),
                    int(min_y.y * image.shape[0]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 240, 15), 2, cv2.LINE_AA)

        cv2.imshow("POC", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

