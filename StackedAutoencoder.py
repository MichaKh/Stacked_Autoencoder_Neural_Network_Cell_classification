from __future__ import print_function
import keras
import numpy
from keras.models import Sequential
from keras.layers.core import *
from sklearn.cross_validation import train_test_split


class StackedAutoencoder(object):
    """
    Implementation of stacked autoencoder multi-class classifier using the Keras Python package.
    This classifier is used to classify cells to cell cycle phases S, G1 or G2M.
    """
    def __init__(self, X, Y, label_encoder, num_labels):
        self.X = X
        self.Y = keras.utils.to_categorical(Y)
        self.auto_encoder = None
        self.encoding_dim = num_labels
        self.label_encoder = label_encoder

        # fix random seed for reproducibility
        self.seed = 7
        numpy.random.seed(7)

    def create_autoencoder(self):
        """
        Build the stacked auto-encoder using multiple hidden layers.
        The stacked auto-encoder is then trained and weights are freezed afterwards.
        A softmax classification layer is that appended to the last layer, replacing the input
        re-constructed layer of the auto-encoder.
        :return: Compiled classification neural network model.
        """
        self.auto_encoder = Sequential()
        self.auto_encoder.add(Dense(3000, activation='relu', input_dim=self.X.shape[1]))
        self.auto_encoder.add(Dense(1000, activation='relu'))
        self.auto_encoder.add(Dense(30, activation='relu'))

        self.auto_encoder.add(Dense(3000, activation='relu'))
        self.auto_encoder.add(Dense(self.X.shape[1], activation='sigmoid'))

        self.auto_encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.auto_encoder.fit(self.X, self.X,
                              epochs=10,
                              batch_size=5,
                              shuffle=True,
                              validation_split=0.33,
                              validation_data=None)

        self.auto_encoder.layers.pop()
        self.auto_encoder.add(Dense(self.encoding_dim, activation='softmax'))
        self.auto_encoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.auto_encoder.summary())

        # Freeze all weights after training the stacked auto-encoder and all the classification layer
        for i in range(0, len(self.auto_encoder.layers)-1):
            self.auto_encoder.layers[i].trainable = False

        return self.auto_encoder

    def evaluate_autoencoder(self):
        """
        Fit the trained neural network and validate it using splitting the dataset to training and testing sets.
        :return: Accuracy score of the classification.
        """
        self.auto_encoder.fit(self.X, self.Y,
                              epochs=10,
                              batch_size=5,
                              shuffle=True)

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.33, random_state=self.seed)
        #predictions = self.auto_encoder.predict_classes(X_test)
        #print(predictions)
        #print(self.label_encoder.inverse_transform(predictions))
        score = self.auto_encoder.evaluate(X_test, Y_test, batch_size=5, verbose=1)
        return score
