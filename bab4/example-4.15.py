# Example 4.15
# https://pypi.org/project/image-classifiers/
# Install package separately if needed:
# pip install image-classifiers

import cv2
import numpy as np

try:
    from classification_models.tfkeras import Classifiers
    from keras.applications.imagenet_utils import decode_predictions
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This example requires the 'image-classifiers' package and TensorFlow/Keras. "
        "Install them with: pip install image-classifiers tensorflow-cpu opencv-python"
    ) from exc

# Set up a model
clf, preprocess_input = Classifiers.get('vgg16')
# clf, preprocess_input = Classifiers.get('resnet50')
# clf, preprocess_input = Classifiers.get('mobilenetv2')
# clf, preprocess_input = Classifiers.get('densenet201')
# clf, preprocess_input = Classifiers.get('inceptionv3')

sz = 224
# sz = 299
model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
model.summary()

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok or cam_frame is None:
        break

    frame = cv2.resize(cam_frame, (sz, sz))
    image = np.asarray(frame)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    preds = model.predict(image)
    label = decode_predictions(preds, top=1)

    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100
    cv2.putText(
        cam_frame,
        f"{class_name}, {confidence:.1f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Classification", cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()