import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Capture Webcam
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw Landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Count Fingers
    count = 0

    if results.multi_hand_landmarks:
      hand_landmarks = list((lm.x,lm.y,) for lm in results.multi_hand_landmarks[0].landmark)
      if results.multi_handedness[0].classification[0].label == "Left" and hand_landmarks[4][0] > hand_landmarks[3][0]:       #Right Thumb
        count = count+1
      elif results.multi_handedness[0].classification[0].label == "Right" and hand_landmarks[4][0] < hand_landmarks[3][0]:       #Left Thumb
        count = count+1
      for i in range(2,6):
        if hand_landmarks[4*i][1] < hand_landmarks[(4*i)-2][1]:      #Other fingers
          count = count+1

    cv2.putText(image, str(count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    cv2.imshow('MediaPipe Hands',image) #, cv2.flip(image, 1))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
