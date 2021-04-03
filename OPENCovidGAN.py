#! /usr/bin/python3

from __future__ import print_function, division

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Multiply, concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, tanh, linear
from tensorflow import keras
from tensorflow.keras.utils import Progbar
import tensorflow as tf

from collections import defaultdict
import pickle as pickle
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

np.random.seed(1337)

class ACGAN():
    def __init__(self):

        self.img_rows = 112
        self.img_cols = 112
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2
        self.latent_dim = 20000

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])

        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([label,noise])

        self.discriminator.trainable = False

        valid, target_label = self.discriminator(img)

        self.combined = Model([label,noise], [valid, target_label])
        self.combined.compile(loss=losses,optimizer=optimizer)

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')

        noise_branch = Dense(1024*7*7)(noise)
        noise_branch = relu(noise_branch)
        noise_branch = Reshape((7, 7, 1024))(noise_branch)
        noise_branch = Model(inputs=noise, outputs=noise_branch)

        label_branch = Embedding(input_dim=50,output_dim=1)(label)
        label_branch = Dense(49,input_shape=(7,7))(label_branch)
        label_branch = linear(label_branch)
        label_branch = Reshape((7, 7, 1),)(label_branch)
        label_branch = Model(inputs=label, outputs=label_branch)

        combined = concatenate([noise_branch.output, label_branch.output])

        combined = Conv2DTranspose(512, (5,5), strides=(2,2),padding="same")(combined)
        combined = BatchNormalization(momentum=0)(combined)
        combined = relu(combined)

        combined = Conv2DTranspose(256, (5,5), strides=(2,2),padding="same")(combined)
        combined = BatchNormalization(momentum=0)(combined)
        combined = relu(combined)

        combined = Conv2DTranspose(128, (5,5), strides=(2,2),padding="same")(combined)
        combined = BatchNormalization(momentum=0)(combined)
        combined = relu(combined)

        combined = Conv2DTranspose(3, (5,5), strides=(2,2),padding="same")(combined)
        combined = tanh(combined)


        model = Model(inputs=[label_branch.input, noise_branch.input], outputs=combined)

        keras.utils.plot_model(model, "generateur.png", show_shapes=True)

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, kernel_size=(3,3), strides=(2,2), padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(256, kernel_size=(3,3), strides=(2,2), padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(512, kernel_size=(3,3), strides=(2,2), padding="same"))
        model.add(BatchNormalization(momentum=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Flatten())

        img = Input(shape=self.img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        keras.utils.plot_model(model, "discriminateur.png", show_shapes=True)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128):

        cxr_train = keras.preprocessing.image_dataset_from_directory("train",labels="inferred",batch_size=2084,image_size=(112, 112))
        cxr_test = keras.preprocessing.image_dataset_from_directory("test",labels="inferred",batch_size=1800,image_size=(112, 112))

        cxr_train_images = []
        cxr_train_labels = []
        cxr_test_images = []
        cxr_test_labels = []

        for images, labels in cxr_train:
            for i in range(len(images)):
              cxr_train_images.append(images[i])
              cxr_train_labels.append(labels[i])

        for images, labels in cxr_test:
            for i in range(len(images)):
              cxr_test_images.append(images[i])
              cxr_test_labels.append(labels[i])

        cxr_train_images = np.array(cxr_train_images)
        cxr_train_labels = np.array(cxr_train_labels)
        cxr_test_images = np.array(cxr_test_images)
        cxr_test_labels = np.array(cxr_test_labels)

        X_train = (cxr_train_images.astype(np.float32) - 127.5) / 127.5

        X_test = (cxr_test_images.astype(np.float32) - 127.5) / 127.5

        nb_train, nb_test = X_train.shape[0], X_test.shape[0]

        train_history = defaultdict(list)
        test_history = defaultdict(list)

        y_train = cxr_train_labels
        y_test = cxr_test_labels

        for epoch in range(epochs):

            print('Epoch {} of {}'.format(epoch + 1, epochs))
            nb_batches = int(X_train.shape[0] / batch_size)
            progress_bar = Progbar(target=nb_batches)


            epoch_gen_loss = []
            epoch_disc_loss = []

            for index in range(nb_batches):
                progress_bar.update(index)

                noise = np.random.normal(0, 0.02, (batch_size, self.latent_dim))

                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                label_batch = y_train[index * batch_size:(index + 1) * batch_size]

                sampled_labels = np.random.randint(0, 1, batch_size)

                generated_images = self.generator.predict([sampled_labels.reshape((-1, 1)),noise], verbose=0)

                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)

                aux_y = np.concatenate((label_batch, sampled_labels))

                epoch_disc_loss.append(self.discriminator.train_on_batch(X, [y, aux_y]))

                noise = np.random.normal(0, 0.02, (2 * batch_size, self.latent_dim))
                sampled_labels = np.random.randint(0, 1, 2 * batch_size)

                trick = np.ones(2 * batch_size)

                epoch_gen_loss.append(self.combined.train_on_batch([sampled_labels.reshape((-1, 1)), noise], [trick, sampled_labels]))

            print('\nTesting for epoch {}:'.format(epoch + 1))
            noise = np.random.normal(0, 0.02, (nb_test, self.latent_dim))

            sampled_labels = np.random.randint(0, 1, nb_test)
            generated_images = self.generator.predict([sampled_labels.reshape((-1, 1)), noise], verbose=False)

            X = np.concatenate((X_test, generated_images))
            y = np.array([1] * nb_test + [0] * nb_test)
            aux_y = np.concatenate((y_test, sampled_labels), axis=0)

            discriminator_test_loss = self.discriminator.evaluate(X, [y, aux_y], verbose=False)

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

            noise = np.random.normal(0, 0.02, (2 * nb_test, self.latent_dim))
            sampled_labels = np.random.randint(0, 1, 2 * nb_test)

            trick = np.ones(2 * nb_test)

            generator_test_loss = self.combined.evaluate([sampled_labels.reshape((-1, 1)),noise],[trick, sampled_labels], verbose=False)

            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

            train_history['generator'].append(generator_train_loss)
            train_history['discriminator'].append(discriminator_train_loss)

            test_history['generator'].append(generator_test_loss)
            test_history['discriminator'].append(discriminator_test_loss)

            print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *self.discriminator.metrics_names))
            print('-' * 65)

            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
            print(ROW_FMT.format('generator (train)',*train_history['generator'][-1]))
            print(ROW_FMT.format('generator (test)',*test_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',*train_history['discriminator'][-1]))
            print(ROW_FMT.format('discriminator (test)',*test_history['discriminator'][-1]))

            if epoch > 2000:
                self.generator.save('saved_model/generator_epoch_{0:03d}.hdf5'.format(epoch))

            r, c = 2, 2
            noise = np.random.normal(0, 0.02, (r * c, self.latent_dim))

            sampled_labels = np.array([num for _ in range(r) for num in range(c)])
            gen_imgs = self.generator.predict([sampled_labels, noise])
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("images/%d.png" % epoch)
            plt.close()

        pickle.dump({'train': train_history, 'test': test_history},open('acgan-history.pkl', 'wb'))



if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=2200, batch_size=64)
