import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def region_movement(prev_gray, gray, x, y, w, h):
    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_now = gray[y:y+h, x:x+w]
    diff = cv2.absdiff(roi_prev, roi_now)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    movement = cv2.countNonZero(thresh)
    return movement

# โหลดโมเดล
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

EAR_THRESHOLD = 0.25
cap = cv2.VideoCapture(1)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    (h, w) = frame_resized.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_resized, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    body_frame = frame_resized.copy()
    blood_frame = frame_resized.copy()
    parts_status = {}

    found_person = False  # ตัวแปรใหม่ไว้ตรวจสอบว่าพบร่างกายหรือไม่

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # person
                found_person = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                box_width = endX - startX
                box_height = endY - startY

                top_height = int(box_height * 0.6)
                bottom_height = box_height - top_height

                left_arm = (startX, startY, box_width // 2, top_height)
                right_arm = (startX + box_width // 2, startY, box_width - box_width // 2, top_height)
                left_leg = (startX, startY + top_height, box_width // 2, bottom_height)
                right_leg = (startX + box_width // 2, startY + top_height, box_width - box_width // 2, bottom_height)

                parts = {
                    "Left Arm": left_arm,
                    "Right Arm": right_arm,
                    "Left Leg": left_leg,
                    "Right Leg": right_leg,
                }

                for part_name, (px, py, pw, ph) in parts.items():
                    movement = region_movement(prev_gray, gray, px, py, pw, ph)
                    status = "Moving" if movement > 200 else "Still"
                    parts_status[part_name] = status
                    color = (0, 0, 255) if status == "Moving" else (0, 255, 0)
                    cv2.rectangle(body_frame, (px, py), (px + pw, py + ph), color, 2)
                    cv2.putText(body_frame, f"{part_name}: {status}", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.rectangle(body_frame, (startX, startY), (endX, endY), (255, 255, 255), 1)

    if not found_person:
        cv2.putText(body_frame, "Patient not detected!", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    face_frame = frame_resized.copy()
    eye_frame = frame_resized.copy()
    faces = detector(gray)

    for face in faces:
        x, y, w_face, h_face = face.left(), face.top(), face.width(), face.height()
        movement_face = region_movement(prev_gray, gray, x, y, w_face, h_face)
        status_face = "Moving" if movement_face > 300 else "Still"
        color_face = (0, 0, 255) if status_face == "Moving" else (0, 255, 0)
        cv2.rectangle(face_frame, (x, y), (x + w_face, y + h_face), color_face, 2)
        cv2.putText(face_frame, status_face, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_face, 1)

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEB = shape[lebStart:lebEnd]
        rightEB = shape[rebStart:rebEnd]
        x_eb, y_eb, w_eb, h_eb = cv2.boundingRect(np.concatenate([leftEB, rightEB]))
        movement_eb = region_movement(prev_gray, gray, x_eb, y_eb, w_eb, h_eb)
        status_eb = "Moving" if movement_eb > 150 else "Still"

        mouth = shape[mStart:mEnd]
        x_m, y_m, w_m, h_m = cv2.boundingRect(mouth)
        movement_mouth = region_movement(prev_gray, gray, x_m, y_m, w_m, h_m)
        status_mouth = "Moving" if movement_mouth > 150 else "Still"

        cv2.drawContours(eye_frame, [cv2.convexHull(leftEye)], -1, (0, 255, 255), 1)
        cv2.drawContours(eye_frame, [cv2.convexHull(rightEye)], -1, (0, 255, 255), 1)
        cv2.drawContours(eye_frame, [cv2.convexHull(leftEB)], -1, (255, 255, 0), 1)
        cv2.drawContours(eye_frame, [cv2.convexHull(rightEB)], -1, (255, 255, 0), 1)
        cv2.drawContours(eye_frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

        cv2.putText(eye_frame, f"EAR: {ear:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(eye_frame, f"Eye: {'Moving' if ear < EAR_THRESHOLD else 'Still'}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0), 2)
        cv2.putText(eye_frame, f"Eyebrow: {status_eb}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if status_eb == "Moving" else (0, 255, 0), 2)
        cv2.putText(eye_frame, f"Mouth: {status_mouth}", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if status_mouth == "Moving" else (0, 255, 0), 2)

    # ตรวจจับเลือด
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
            cv2.rectangle(blood_frame, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 0, 255), 2)
            cv2.putText(blood_frame, "Blood Detected", (x_r, y_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # แสดงผล
    if parts_status:
        status_text = ", ".join([f"{part}: {status}" for part, status in parts_status.items()])
        cv2.putText(body_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Person Detection', body_frame)
    cv2.imshow('Face Detection', face_frame)
    cv2.imshow('Eye, Eyebrow, Mouth Detection', eye_frame)
    cv2.imshow('Blood Detection', blood_frame)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
