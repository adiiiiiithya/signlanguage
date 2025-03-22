import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #convert2rgb
        results = hands.process(frame_rgb)  #detectionofhand

        if results.multi_hand_landmarks:  
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #drawsonhandinrealtime

                #thumb4index8middle12ring16pinky20wrist0
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]
                wrist = hand_landmarks.landmark[0]

                
                gesture = "Unknown" #default

                #lowerymeanshigherup
                if (
                    index_tip.y < hand_landmarks.landmark[6].y and
                    middle_tip.y < hand_landmarks.landmark[10].y and
                    ring_tip.y < hand_landmarks.landmark[14].y and
                    pinky_tip.y < hand_landmarks.landmark[18].y
                ):
                    gesture = "Open Hand"

            
                elif (
                    thumb_tip.y < wrist.y and
                    index_tip.y > hand_landmarks.landmark[6].y and
                    middle_tip.y > hand_landmarks.landmark[10].y and
                    ring_tip.y > hand_landmarks.landmark[14].y and
                    pinky_tip.y > hand_landmarks.landmark[18].y
                ):
                    gesture = "Thumbs Up"

                #iloveyou
                elif (
                    thumb_tip.y < wrist.y and
                    index_tip.y < hand_landmarks.landmark[6].y and
                    pinky_tip.y < hand_landmarks.landmark[18].y and
                    middle_tip.y > hand_landmarks.landmark[10].y and
                    ring_tip.y > hand_landmarks.landmark[14].y
                ):
                    gesture = "I Love You"

                
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", frame)  
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  #pressqtoquit
            break

cap.release()
cv2.destroyAllWindows()
