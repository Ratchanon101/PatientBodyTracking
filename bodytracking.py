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

    # ----- 3. Eye detection + EAR -----
    eye_frame = frame_resized.copy()
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(eye_frame, [leftHull], -1, (0, 255, 255), 1)
        cv2.drawContours(eye_frame, [rightHull], -1, (0, 255, 255), 1)

        # Check eye region movement
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        roi = diff_thresh[y:y+h, x:x+w]
        movement = cv2.countNonZero(roi)
        movement_status = "Moving" if movement > 200 else "Still"

        cv2.putText(eye_frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(eye_frame, movement_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if movement_status == "Moving" else (0, 255, 0), 1)

    # แสดงผลแยก 3 หน้าต่าง
    cv2.imshow('Body Detection', body_frame)
    cv2.imshow('Face Detection', face_frame)
    cv2.imshow('Eye Detection + EAR', eye_frame)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
