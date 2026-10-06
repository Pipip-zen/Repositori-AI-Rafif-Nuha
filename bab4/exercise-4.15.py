# Example 4.17 Modified: CNN Visualize Filters for Layer 2 (block1_conv2)
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
# summarize the model
model.summary()
n = 0
for layer in model.layers:
    print(n, layer.name)
    n += 1

n = 0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)
    else:
        print(n, layer.name)
    n += 1

# retrieve weights from the second convolutional layer (n = 2)
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 4, 1
pyplot.figure(figsize=(10, 10))
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

# load the model for feature map extraction
model = VGG19()
# Select the hidden layer to visualize (n = 2)
n = 2
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
model.summary()

# Download a sample elephant image using a User-Agent header to prevent 403 Forbidden
url = 'https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg'
req = urllib.request.Request(
    url,
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
)
with urllib.request.urlopen(req) as response, open('Elephant.jpg', 'wb') as out_file:
    out_file.write(response.read())

# load the image with the required shape
img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# get feature map for the second hidden layer
feature_maps = model.predict(img)
print("Feature maps: ", feature_maps.shape)
plot_feature_maps(feature_maps)