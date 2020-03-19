"""
module that runs face detection and alignment algorithm.
Run command -> python video.py
"""
import cv2
from detector_aligner import DetectorAligner

# init our class
detector_aligner = DetectorAligner(use_hog=True, desiredFaceWidth=512)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations to find and align faces
    face_list = detector_aligner(frame)
    # draw boxes on frame and print it then
    for i, face in enumerate(face_list):
        cv2.imshow(f'Face #{i + 1}', face)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
