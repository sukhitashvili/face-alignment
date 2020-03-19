import dlib
import cv2
from facealigner import FaceAligner

DETECTOR_MODEL_PATH = 'dlib_models/mmod_human_face_detector.dat'
LANDMATK_MODEL_PATH = 'dlib_models/shape_predictor_68_face_landmarks.dat'


class DetectorAligner:
    def __init__(self,
                 use_hog=True,
                 desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=512,
                 desiredFaceHeight=None):
        """
        Detects and aligns face on image.
        Args:
            use_hog: flag whether use dlib hog or cnn for face detection.
            desiredLeftEye : An optional (x, y) tuple with the default shown, specifying the desired output left eye position. For this variable, it is common to see percentages within the range of 20-40%. These percentages control how much of the face is visible after alignment. The exact percentages used will vary on an application-to-application basis. With 20% you’ll basically be getting a “zoomed in” view of the face, whereas with larger values the face will appear more “zoomed out.”
            desiredFaceWidth : Another optional parameter that defines our desired face with in pixels. We default this value to 256 pixels.
            desiredFaceHeight: The final optional parameter specifying our desired face height value in pixels. If None then set to width value.
        """
        if use_hog:
            self.detector = dlib.get_frontal_face_detector()
        else:
            self.detector = dlib.cnn_face_detection_model_v1(DETECTOR_MODEL_PATH)

        self.predictor = dlib.shape_predictor(LANDMATK_MODEL_PATH)
        self.align_face = FaceAligner(self.predictor,
                                      desiredLeftEye=desiredLeftEye,
                                      desiredFaceWidth=desiredFaceWidth,
                                      desiredFaceHeight=desiredFaceHeight)

    def __call__(self, image):
        """
        Detects faces and aligns them.
        Args:
            image: numpy nd-array of image.

        Returns: python list of aligned faces of specified height and width
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 2)
        # loop over the face detections
        faces = []
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            faceAligned = self.align_face.align(image, gray, rect)
            faces.append(faceAligned)

        return faces
