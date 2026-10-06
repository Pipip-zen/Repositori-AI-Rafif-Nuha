# Example 4.13
import cv2
import numpy as np

try:
    from keras.applications import vgg16
    from keras.applications.imagenet_utils import decode_predictions
    from keras.preprocessing.image import img_to_array
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This example requires TensorFlow/Keras with a working backend. "
        "Install it with: pip install tensorflow-cpu"
    ) from exc

image_size = 224
model = vgg16.VGG16(weights='imagenet')
print(model.summary())

camera = cv2.VideoCapture(0)
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok or cam_frame is None:
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    label = decode_predictions(predictions, top=1)
    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100

    cv2.putText(
        cam_frame,
        f"VGG16: {class_name}, {confidence:.1f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()