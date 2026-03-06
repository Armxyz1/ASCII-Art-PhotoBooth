import mediapipe as mp
import cv2
import numpy as np

mp_mesh = mp.solutions.face_mesh


def align_face(face_img):

    with mp_mesh.FaceMesh(static_image_mode=True) as mesh:

        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        results = mesh.process(rgb)

        if not results.multi_face_landmarks:
            return face_img

        landmarks = results.multi_face_landmarks[0]

        h, w, _ = face_img.shape

        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]

        lx, ly = int(left_eye.x*w), int(left_eye.y*h)
        rx, ry = int(right_eye.x*w), int(right_eye.y*h)

        dy = ry - ly
        dx = rx - lx

        angle = np.degrees(np.arctan2(dy, dx))

        center = (w//2, h//2)

        M = cv2.getRotationMatrix2D(center, angle, 1)

        aligned = cv2.warpAffine(face_img, M, (w, h))

        return aligned