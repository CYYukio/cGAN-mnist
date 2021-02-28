import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Embedding, Flatten, Conv2D, Dense

class cGAN():
    def __init__(self):
        # 参数
        self.img_height = 28
        self.img_width = 28
        self.img_channel = 1
        self.img_shape = (self.img_width, self.img_height, self.img_channel)
        self.class_num = 10
        self.latent_dim = 100

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        # 判别器
        self.discriminator = self.define_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 生成器
        self.generator = self.define_generator()

        noise = keras.layers.Input(shape=(self.latent_dim,))
        label = keras.layers.Input(shape=(1,))  # 手写字标签就一个（0-9）

        gen_img = self.generator([noise, label])

        self.discriminator.trainable = False

        # 训练generator，标签即为判别器的评价
        valid = self.discriminator([gen_img, label])
        self.combined = keras.Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def define_discriminator(self):
        model = keras.models.Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=self.img_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(keras.layers.Dropout(0.4))  # 提高泛化
        model.add(Dense(units=1, activation="sigmoid"))
        model.summary()

        img = keras.layers.Input(shape=self.img_shape)
        label = keras.layers.Input(shape=(1,), dtype='int32')

        label_embedding = Embedding(self.class_num, np.prod(self.img_shape))(label)
        n_nodes = self.img_height*self.img_width
        label_embedding = Dense(n_nodes)(label_embedding)
        label_embedding = keras.layers.Reshape((self.img_width, self.img_height, 1))(label_embedding)

        model_input = keras.layers.multiply([img, label_embedding])
        valid2 = model(model_input)

        #之后再更新判别器
        return keras.models.Model([img, label], valid2)

    def define_generator(self):
        model = keras.models.Sequential()
        n_nodes = 256 * 3 * 3
        model.add(keras.layers.Dense(n_nodes, input_dim=self.latent_dim))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # 调整输出形状（3，3，256）
        model.add(keras.layers.Reshape((3, 3, 256)))
        # 上采样到8*8
        model.add(keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='valid'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # 上采样到16*16
        model.add(keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # 上采样到32*32
        model.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # output
        model.add(keras.layers.Conv2D(1, (3, 3), activation='tanh', padding='same'))
        model.add(keras.layers.Reshape(self.img_shape))
        model.summary()

        noise = keras.layers.Input(shape=self.latent_dim)
        label = keras.layers.Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.class_num, np.prod(self.img_shape))(label))

        n_nodes = 100
        label_embedding = Dense(n_nodes)(label_embedding)

        label_embedding = keras.layers.Reshape((self.latent_dim, ))(label_embedding)

        model_input = keras.layers.multiply([noise, label_embedding])
        img = model(model_input)
        return keras.models.Model([noise, label], img)

    def train(self, epochs, batch_size=128):
        (X_train, y_train), (_, _) = keras.datasets.mnist.load_data()
        #归一到-1 - 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)
        #行变列

        true = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            #训练判别器
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))
            #生成器生成假图
            gen_imgs = self.generator.predict([noise, labels])

            #训练
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], true)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5*np.add(d_loss_fake,d_loss_real)


            ##训练生成器
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], true)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        filename = 'cGAN_generator_model.h5'
        self.generator.save(filename)


if __name__ == '__main__':
    cgan = cGAN()
    cgan.train(epochs=10000, batch_size=32)

