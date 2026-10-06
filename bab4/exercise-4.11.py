# VGG19 Webcam Real-time Classification Program
import cv2
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.imagenet_utils import decode_predictions

# Load VGG19 model with ImageNet weights
model = VGG19(weights='imagenet')

# Initialize webcam capture (index 0)
cap = cv2.VideoCapture(0)

print("Memulai webcam... Tekan 'q' untuk keluar.")

while True:
  ret, frame = cap.read()
  if not ret:
    break

  # Ubah ukuran frame ke (224, 224) sesuai input VGG19
  resized_frame = cv2.resize(frame, (224, 224))
  x = np.expand_dims(resized_frame, axis=0)
  x = preprocess_input(x)

  # Prediksi objek pada frame
  predictions = model.predict(x, verbose=0)
  results = decode_predictions(predictions, top=1)[0]
  _, label, score = results[0]

  # Tampilkan label teks pada jendela video
  text = f"{label}: {score*100:.2f}%"
  cv2.putText(
      frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
  )
  cv2.imshow('VGG19 Webcam Classification', frame)

  # Tekan tombol 'q' untuk menghentikan program
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()