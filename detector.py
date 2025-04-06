import cv2
import mediapipe as mp
import math


cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)   


mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)  

#la malla facial
mpMallaFacial = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) 


lista = []

