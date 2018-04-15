from __future__ import print_function
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class OrdinalClassifier(object):
    """
    Implementation of ordinal binary classifier using the Keras Python package.
    This classifier is used to classify phases G1 vs. S+G2M and G1+S vs. G2M.
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        # fix random seed for reproducibility
        self.seed = 7
        numpy.random.seed(7)

    def create_classifier(self):
        """
        Create the sequential structure of the neural network (input, output and hidden layers)
        :return: Model of the staked neural network
        """
        # create model
        model = Sequential()
        model.add(Dense(1024, input_shape=(self.X.shape[1],), kernel_initializer='normal', activation='relu'))
        model.add(Dense(512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def classify(self):
        """
        Apply a stratified Keras k-fold classifier on the trained neural network.
        :return: The accuracy and standard deviation of the  k-fold cross validation on the dataset.
        """
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=self.create_classifier, epochs=10, batch_size=5, verbose=1)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        results = cross_val_score(pipeline, self.X, self.Y, cv=kfold)
        print("Smaller: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        return results.mean(), results.std()
