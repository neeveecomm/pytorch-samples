import cv2
fps = 25
width = 640
height = 480
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height), cv2.IMREAD_COLOR)

x = 0
while x < 100:
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('video demo', frame)
    x = x + 1
    if cv2.waitKey(10) & 0xFF == 'q':
        break

out.release()
cap.release()
cv2.destroyAllWindows()