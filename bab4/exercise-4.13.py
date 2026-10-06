# Example 4.15 Modified: Comparing Multiple Models via image-classifiers
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

# Pilih model yang ingin diuji: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', atau 'inceptionv3'
model_name = 'mobilenetv2' 
clf, preprocess_input = Classifiers.get(model_name)

# Sesuaikan ukuran input (InceptionV3 umumnya menggunakan 299, lainnya 224)
sz = 299 if model_name == 'inceptionv3' else 224
model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
print(f"Menjalankan model: {model_name.upper()}")
model.summary()

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok or cam_frame is None:
        break

    frame = cv2.resize(cam_frame, (sz, sz))
    image = np.asarray(frame, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    preds = model.predict(image, verbose=0)
    label = decode_predictions(preds, top=1)

    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100
    cv2.putText(
        cam_frame,
        f"[{model_name.upper()}] {class_name}, {confidence:.1f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Model Comparison - Image Classifiers", cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # Tekan 'ESC' untuk keluar
        break

camera.release()
cv2.destroyAllWindows()