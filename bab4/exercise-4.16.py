# Example 4.17 Modified: Display All Filters for Layer (n = 2)
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
from numpy import expand_dims
import urllib.request

# load the model
model = VGG19()
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot all filters by setting n_filters to s[3] (total number of filters)
n_filters, ix = s[3], 1
pyplot.figure(figsize=(12, 12))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[j, :, :], cmap='gray')
        ix += 1
pyplot.show()

def plot_feature_maps(feature_maps):
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()

# load model for feature map extraction
model = VGG19()
model = Model(inputs=model.inputs, outputs=model.layers[n].output)

# Download and load sample image
url = 'https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg'
req = urllib.request.Request(
    url,
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
)
with urllib.request.urlopen(req) as response, open('Elephant.jpg', 'wb') as out_file:
    out_file.write(response.read())

img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

feature_maps = model.predict(img)
print("Feature maps: ", feature_maps.shape)
plot_feature_maps(feature_maps)