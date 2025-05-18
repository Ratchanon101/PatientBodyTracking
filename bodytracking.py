import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EAR_THRESHOLD = 0.25

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # ----- 1. Body detection -----
    body_frame = frame_resized.copy()
    body_diff = diff_thresh.copy()
    boxes, _ = hog.detectMultiScale(body_frame, winStride=(8, 8))
    for (x, y, w, h) in boxes:
        roi = body_diff[y:y+h, x:x+w]
        movement = cv2.countNonZero(roi)
        status = "Moving" if movement > 500 else "Still"
        color = (0, 0, 255) if status == "Moving" else (0, 255, 0)
        cv2.rectangle(body_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(body_frame, status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ----- 2. Face detection -----
    face_frame = frame_resized.copy()
    face_diff = diff_thresh.copy()
    faces = detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        roi = face_diff[y:y+h, x:x+w]
        movement = cv2.countNonZero(roi)
        status = "Moving" if movement > 300 else "Still"
        color = (0, 0, 255) if status == "Moving" else (0, 255, 0)
        cv2.rectangle(face_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(face_frame, status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ----- 3. Eye, Eyebrow, Mouth detection + EAR -----
    eye_frame = frame_resized.copy()
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # ตา
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(eye_frame, [leftHull], -1, (0, 255, 255), 1)
        cv2.drawContours(eye_frame, [rightHull], -1, (0, 255, 255), 1)

        # คิ้ว
        leftEyebrow = shape[17:22]
        rightEyebrow = shape[22:27]
        leftEyebrowHull = cv2.convexHull(leftEyebrow)
        rightEyebrowHull = cv2.convexHull(rightEyebrow)
        cv2.drawContours(eye_frame, [leftEyebrowHull], -1, (255, 255, 0), 1)
        cv2.drawContours(eye_frame, [rightEyebrowHull], -1, (255, 255, 0), 1)

        # ปาก
        mouth = shape[48:68]
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(eye_frame, [mouthHull], -1, (0, 255, 0), 1)

        # คำนวณ movement ของแต่ละส่วน
        # Eye ROI
        ex, ey, ew, eh = cv2.boundingRect(leftEye)
        eye_roi_left = diff_thresh[ey:ey+eh, ex:ex+ew]
        movement_eye_left = cv2.countNonZero(eye_roi_left)

        ex, ey, ew, eh = cv2.boundingRect(rightEye)
        eye_roi_right = diff_thresh[ey:ey+eh, ex:ex+ew]
        movement_eye_right = cv2.countNonZero(eye_roi_right)

        movement_eye = movement_eye_left + movement_eye_right
        eye_status = "Moving" if movement_eye > 50 else "Still"

        # Eyebrow ROI
        ex, ey, ew, eh = cv2.boundingRect(leftEyebrow)
        eyebrow_roi_left = diff_thresh[ey:ey+eh, ex:ex+ew]
        movement_eyebrow_left = cv2.countNonZero(eyebrow_roi_left)

        ex, ey, ew, eh = cv2.boundingRect(rightEyebrow)
        eyebrow_roi_right = diff_thresh[ey:ey+eh, ex:ex+ew]
        movement_eyebrow_right = cv2.countNonZero(eyebrow_roi_right)

        movement_eyebrow = movement_eyebrow_left + movement_eyebrow_right
        eyebrow_status = "Moving" if movement_eyebrow > 30 else "Still"

        # Mouth ROI
        ex, ey, ew, eh = cv2.boundingRect(mouth)
        mouth_roi = diff_thresh[ey:ey+eh, ex:ex+ew]
        movement_mouth = cv2.countNonZero(mouth_roi)
        mouth_status = "Moving" if movement_mouth > 50 else "Still"

        # แสดงสถานะบนหน้าต่าง
        cv2.putText(eye_frame, f"EAR: {ear:.2f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(eye_frame, f"Eye: {eye_status}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if eye_status == "Moving" else (0, 255, 0), 1)
        cv2.putText(eye_frame, f"Eyebrow: {eyebrow_status}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if eyebrow_status == "Moving" else (0, 255, 0), 1)
        cv2.putText(eye_frame, f"Mouth: {mouth_status}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if mouth_status == "Moving" else (0, 255, 0), 1)

    # ----- 4. Blood detection (simple color detection) -----
    blood_frame = frame_resized.copy()
    hsv = cv2.cvtColor(blood_frame, cv2.COLOR_BGR2HSV)

    # กำหนดช่วงสีแดงใน HSV (แดงมี 2 ช่วง hue เพราะ hue วนรอบ 0)
    lower_red1 = (0, 70, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 70, 50)
    upper_red2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # หาจำนวนพิกเซลสีแดง
    red_pixels = cv2.countNonZero(mask)

    if red_pixels > 500:  # กำหนดค่า threshold ตามต้องการ
        # หา contours รอบบริเวณสีแดง
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # กรองเฉพาะพื้นที่ใหญ่ๆ
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(blood_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(blood_frame, "Blood Detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                break

    # แสดงผลแยก 4 หน้าต่าง
    cv2.imshow('Body Detection', body_frame)
    cv2.imshow('Face Detection', face_frame)
    cv2.imshow('Eye, Eyebrow, Mouth Detection + EAR', eye_frame)
    cv2.imshow('Blood Detection', blood_frame)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
