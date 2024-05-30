import tensorflow as tf
import cv2 as cv
import mediapipe as mp
import numpy as np
import time, os

#['heart', 'iloveu', 'smile', 'fine', 'live', 'what', 'best']
actions = ['best', 'hello', 'what', 'call', 'happy', 
           'iloveyou', 'see', 'smile', 'peace', 'me', 
           'meet', 'heart', 'fine']
seq_len = 30
secs_for_action = 30

capture = cv.VideoCapture(0)
time_created = int(time.time())

mp_hands = mp.solutions.hands
mp_drawing_utils = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence = 0.5, # 최소 검출 신뢰도
    min_tracking_confidence = 0.5 # 최소 추적 신뢰도
)

os.makedirs('mog_dataset2', exist_ok = True)

# 각 동작별로 데이터 수집 및 저장 수행
while capture.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = capture.read()

        img = cv.flip(img, 1)

        cv.putText(img, f'Waiting for collecting {action.upper()} action...', 
                   org=(10, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv.imshow('img', img)
        cv.waitKey(5000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = capture.read()
            img = cv.flip(img, 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing_utils.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv.imshow('img', img)
            if cv.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('mog_dataset2', f'raw_{action}_{time_created}'), data)

        full_seq_data = []
        for seq in range(len(data) - seq_len):
            full_seq_data.append(data[seq:seq + seq_len])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('mog_dataset2', f'seq_{action}_{time_created}'), full_seq_data)
    break