# Example 4.15 Modified: Model Selection using If-Else Statement
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

# Pilih model yang ingin digunakan: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', atau 'inceptionv3'
model_name = 'resnet50'

# Menggunakan if-else statement untuk memilih model dan ukuran input (sz) yang sesuai
if model_name == 'vgg16':
    clf, preprocess_input = Classifiers.get('vgg16')
    sz = 224
elif model_name == 'resnet50':
    clf, preprocess_input = Classifiers.get('resnet50')
    sz = 224
elif model_name == 'mobilenetv2':
    clf, preprocess_input = Classifiers.get('mobilenetv2')
    sz = 224
elif model_name == 'densenet201':
    clf, preprocess_input = Classifiers.get('densenet201')
    sz = 224
elif model_name == 'inceptionv3':
    clf, preprocess_input = Classifiers.get('inceptionv3')
    sz = 299
else:
    raise ValueError(f"Model '{model_name}' tidak dikenali atau tidak didukung.")

model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
print(f"Menjalankan model: {model_name.upper()} dengan ukuran input {sz}x{sz}")
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
    cv2.imshow("If-Else Model Selection", cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # Tekan 'ESC' untuk keluar
        break

camera.release()
cv2.destroyAllWindows()