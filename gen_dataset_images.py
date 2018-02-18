from keras.datasets import fashion_mnist, mnist
from PIL import Image
import numpy as np

# create an image for Mnist
(x_train, y_train), (_,_) = mnist.load_data()
num_classes = 10
imgs = []
for i in range(num_classes):
    ind = (y_train == i)
    imgs.append(x_train[ind][0])
img = np.concatenate(imgs, axis=-1)
img = Image.fromarray(img)
img.save('pictures/mnist.png')

# create an image for FashionMnist
(x_train, y_train), (_,_) = fashion_mnist.load_data()
num_classes = 10
imgs = []
for i in range(num_classes):
    ind = (y_train == i)
    imgs.append(x_train[ind][0])
img = np.concatenate(imgs, axis=-1)
img = Image.fromarray(img)
img.save('pictures/fashion_mnist.png')
