import dlib
import cv2
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img = cv2.imread("")
dets = detector(img, 1)
heatmap = np.zeros_like(img, dtype=np.float32)
dets = detector(img, 1)
for d in dets:
    shape = predictor(img, d)
    for index, pt in enumerate(shape.parts()):
        cv2.circle(heatmap, (pt.x, pt.y), 2, (255), 2)
ksize = (15, 15)
heatmap = cv2.GaussianBlur(heatmap, ksize, 0)
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = np.uint8(heatmap)
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
cv2.imshow('Colored Heatmap', colored_heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()