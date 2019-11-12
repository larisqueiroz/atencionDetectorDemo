import cv2
import dlib
import time

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
t = time.time()
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    faces = detector(gray)

    detect = detector(gray)

    for face in faces:
        x1r = face.left()
        y1r = face.top()
        x2r = face.right()
        y2r = face.bottom()
        cv2.rectangle(frame, (x1r,y1r), (x2r, y2r), (0,255,0), 3) # face
        roi_color = frame[y1r:y2r, x1r:x2r]
        landmarks = predictor(gray, face)

        for n in range(36,48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #cv2.circle(gray, (x,y), 2, (255, 0, 0), -1) landmarks
            shape=predictor(frame,detect[0])
            x1 = shape.part(36).x
            x2 = shape.part(39).x
            y1 = shape.part(37).y
            y2 = shape.part(40).y
            lefteye = gray[y1-10:y2+10, x1-10:x2+10] # olho esquerdo
            left = frame[y1-10:y2+10, x1-10:x2+10]
            cv2.imshow('left', lefteye)
            right = frame[y1 - 10:y2 + 10, x1 - 10:x2 + 10]
            rowsl, colsl, _ = left.shape
            ret, thresholdl = cv2.threshold(lefteye, 35, 255, cv2.THRESH_BINARY_INV) # selecionando a iris dos olhos
            contoursl, _ = cv2.findContours(thresholdl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contoursl:
                (al, bl, wl, hl) = cv2.boundingRect(cnt)
                cv2.rectangle(right, (al, bl), (al + wl, bl + hl), (255, 0, 0), 1)
                areal = cv2.contourArea(cnt)
                print(f'esquerdo:{areal}')
                if (areal > 165 or areal < 30) and (time.time() -t) > 5: # estimaçao de distraçao em relaçao a area da iris detectada
                    t= time.time()
                    print("DESVIO DE ATENCAO")
                    string = "DESVIO DE ATENCAO"
                cv2.line(right, (al + int(wl / 2), 0), (al + int(wl / 2), rowsl), (0, 255, 0), 1) # linhas dos olhos
                cv2.line(right, (0, bl + int(hl / 2)), (colsl, bl + int(hl / 2)), (0, 255, 0), 1)
                break


            x11 = shape.part(42).x
            x22 = shape.part(45).x
            y11 = shape.part(43).y
            y22 = shape.part(46).y
            righteye=gray[y11-10:y22+10,x11-10:x22+10] # olho direito
            right = frame[y11-10:y22+10,x11-10:x22+10]
            rows, cols, _ = right.shape
            ret, threshold = cv2.threshold(righteye, 35, 255, cv2.THRESH_BINARY_INV)
            contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                (a, b, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(right, (a,b), (a+w, b+h), (255,0,0), 1)
                arear = cv2.contourArea(cnt)
                print(f'direito:{arear}')
                if (arear > 165 or arear < 30) and (time.time() -t) > 5: # estimaçao de distraçao em relaçao a area da iris detectada
                    t = time.time()
                    print("DESVIO DE ATENCAO")
                    string = "DESVIO DE ATENCAO"
                    cv2.putText(frame, string, (x1r, y1r), font, 1, (0, 0, 255))
                cv2.line(right, (a+int(w/2), 0), (a+int(w/2),rows), (0,255,0), 1) # linhas dos olhos
                cv2.line(right, (0, b+int(h/2)), (cols, b+int(h/2)), (0, 255,0),1)
                break

            cv2.imshow('thr', threshold)
            cv2.imshow('thl', thresholdl)
            cv2.imshow('right', righteye)
            cv2.imshow('right', right)
            cv2.imshow('left', left)

    cv2.imshow('frame', frame)

    if cv2.waitKey(30) == ord('q'):
        break

    time.sleep(0.02)

cap.release()
cv2.destroyAllWindows()