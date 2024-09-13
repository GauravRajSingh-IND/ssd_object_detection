import cv2 as cv

# named window.
winName = "Object Detection"
cv.namedWindow(winName)

# create video object
cap = cv.VideoCapture(0) # 0:webcam
if not cap.isOpened():
    raise "error while accessing webcam."

while True:

    # read frame one by one.
    has_frame, frame = cap.read()
    if not has_frame:
        print("No frame to read")
        break

    # display video frame.
    cv.imshow(winName, frame)
    key = cv.waitKey(1)

    # break the loop if user press 'q', 'Q' or esc key.
    if key == ord('q') or key == ord('Q') or key == 27:
        print("Video ended by user.")
        break

cap.release()
cv.destroyAllWindows()




