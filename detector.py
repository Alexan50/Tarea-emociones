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

while True:
    ret, frame = cap.read()  
    if not ret:
        break  
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    resultados = mpMallaFacial.process(frameRGB)

    if resultados.multi_face_landmarks:  # Si detectamos algún rostro
        for rostros in resultados.multi_face_landmarks:
            
            mpDibujo.draw_landmarks(frame, rostros, mp.solutions.face_mesh.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            lista.clear()

            for id, puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x, y = int(puntos.x * an), int(puntos.y * al)
                lista.append([id, x, y])

                if len(lista) == 468:  # Aseguramos que se recojan todos los puntos
                  
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    longitud2 = math.hypot(x4 - x3, y4 - y3)


                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    longitud3 = math.hypot(x6 - x5, y6 - y5)

                    
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    longitud4 = math.hypot(x8 - x7, y8 - y7)

                 
                    if longitud3 > 70:  
                        cv2.putText(frame, 'Persona Feliz', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)


                    
                    if longitud4 > 50:  
                        cv2.putText(frame, 'Persona Asombrada', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    
                    elif longitud1 < 30 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Enojada', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    elif longitud1 > 10 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 80 and longitud3 < 109 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Triste', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Reconocimiento de Emociones", frame)
    t = cv2.waitKey(1)

    if t == 27:  
        break

cap.release()
cv2.destroyAllWindows()