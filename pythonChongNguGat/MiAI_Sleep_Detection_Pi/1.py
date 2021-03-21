import cv2
webcam = cv2.VideoCapture(0)
while True:
	check, frame = webcam.read()
        cv2.imshow("Input", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break