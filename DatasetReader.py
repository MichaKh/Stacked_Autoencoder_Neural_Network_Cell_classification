import numpy as np
import csv
from itertools import izip
import pandas
from sklearn.preprocessing import LabelEncoder


class DatasetReader(object):
    """
    Reads the dataset to dataframe structure and separates the features used for training (X) and the label vector (Y).
    In addition, this class performs normalization and pre-processing.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.X = None
        self.Y = None

    def transpose_data(self):
        """
        Transposes the dataframe tabular object, to access rows as samples and columns as features.
        :return: Writes the transposed dataframe to a CSV format file.
        """
        try:
            # transpose the csv file
            a = izip(*csv.reader(open(self.file_path, "rb")))
            csv.writer(open("transposed_" + self.file_path, "wb")).writerows(a)
        except ValueError:
            print("Cannot access file in specified path!")

    def load_data(self):
        """
        Read the dataset to a dataframe object for further processing, encode the target value and separate the data
         to set of features X and the label vector Y
        :return: Set of features X, label vector Y and the labelEncoder object.
        """
        try:
            # load dataset
            dataframe = pandas.read_csv(self.file_path, index_col=0)
            df = self.preprocess(dataframe)
            dataset = df.values
            # split into input (X) and output (Y) variables
            self.X = dataset[1:-1, 3:].astype(float).T
            self.Y = dataset[-1, 3:].T

            # encoded_Y = self.encode_class()
            label_encoder, encoded_Y = self.encode_class()
            return self.X, encoded_Y, label_encoder
        except ValueError:
            print("Cannot access file in specified path!")

    def encode_class(self):
        """
        Converts the string label value to integer values, to pass the resulting vector for fitting the model.
        :return: Encoded variable Y and its encoder object
        """
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(self.Y)
        encoded_Y = encoder.transform(self.Y)
        return encoder, encoded_Y

    @staticmethod
    def normalize_rpkm(df):
        """
        Normalize the dataframe to reads per kilo milion (RPKM)
        (10^9)*(count_val)/(gene_length*count_sum)
        :param df: Dataframe representing the dataset
        :return:
        """
        # (10^9)*(count_val)/(gene_length*count_sum)
        header = list(df.columns.values)
        for index, row in df.iterrows():
            for col in header[3:]:
                df.loc[index, col] = 10**9 * (row[col])/(row['GeneLength']*df.at['Aligned', col])

    def preprocess(self, df):
        """
        Perform Pre-processing of read counts, including normalization to log2 and removing unexpressed genes.
        :param df: Dataframe representing the dataset.
        :return: Clean and processed dataframe.
        """
        # normalize all read counts to logarithm.
        # remove last row which represent the class label
        label_row = df.tail(1)
        df = df.drop(df.index[len(df) - 1])
        df = self.log_normalize(df)
        df = self.filter_unexpressed_genes(df)
        # append the labels row to the dataframe
        df = df.append(label_row)
        return df

    @staticmethod
    def log_normalize(df):
        """
        Apply logarithm to all values of the dataset
        :param df: Dataframe representing the dataset.
        :return: Normalized dataframe
        """
        cols = df.columns.values
        for col in cols[3:]:
            df[col] = df[col].astype(np.int32)
            df[col] = df[col].apply(lambda x: np.log2(x) if x is not 0 else x)
            # postprocess all -inf values caused by performing log on 0
            # df[col] = df[col].apply(lambda x: 0 if isneginf(x) else x)
        return df

    @staticmethod
    def filter_unexpressed_genes(df):
        """
        Remove all rows (gene features) that have zero count (no expression) in more than 20% of the cells.
        :param df: Dataframe representing the dataset.
        :return: Filtered dataset.
        """
        df = df.loc[~df.apply(lambda row: (row[3:] == 0).all() or
                                          (row != 0).sum() < 0.2*df.shape[1], axis=1)]
        return df


















