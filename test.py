import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Embedding, Flatten, Conv2D, Dense


def create_plot(examples, n, labels):
    # plot images
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        plt.title(labels[i])
        # plot raw pixel data
        plt.imshow(examples[i, :, :], cmap='gray')
    plt.show()


def generate_img():
    # load model
    model = tf.keras.models.load_model('cGAN_generator_model.h5')

    sampled_labels = np.random.randint(0, 10, 32).reshape(-1, 1)
    noise = np.random.normal(0, 1, (32, 100))
    gen_imgs = model.predict([noise, sampled_labels])

    gen_imgs = gen_imgs.reshape(gen_imgs.shape[0], 28, 28)
    create_plot(gen_imgs, 5, sampled_labels)




if __name__ == '__main__':
    generate_img()
