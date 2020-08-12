import cv2

# Initilization of capture
cap = cv2.VideoCapture(0) #'test.mp4')
img = cv2.imread('E:\sample.png', cv2.IMREAD_GRAYSCALE)
while True:
    # readind each frame from capture device
    ret, frame = cap.read()
    cv2.rectangle(frame, (200,150), (400,350), (255, 0, 0))
    cv2.putText(frame, 'video demo', (10,400), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0))
    print ('Size = {} datatype = {}'.format(frame.size, frame.dtype))
    #for x in range(100):
    #    frame[100, x] = [255, 255, 255]

    #frame[10,20:10,20] = [0, 0, 255]
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #test = out + img
    test = cv2.addWeighted(out, 0.9, img, 0.10, 0)
    # showing captured frame
    cv2.imshow('Video demo', test)
    # Wait for a key
    if cv2.waitKey(25) & 0xff == 'q':
        break

# Releasing capture
cap.release()

#destroy all the windows
cv2.destroyAllWindows()
