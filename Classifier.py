from DatasetReader import DatasetReader
from OrdinalClassifier import OrdinalClassifier
from StackedAutoencoder import StackedAutoencoder


def main():
    build_stacked_ae("singlecells_counts.csv")
    build_binary_classifiers("singlecells_counts_G1_vs_SG2M.csv", "singlecells_counts_G1S_vs_G2M.csv")


def build_stacked_ae(path):
    """
    Build the stacked auto-encoder neural network, and evaluate its performance
    :param path: Path to the genetic dataset
    :return: Accuracy of classification of cell cycle phase.
    """
    ############### Stacked Auto-Encoders ##############
    dr = DatasetReader(path)
    train = dr.load_data()
    ae = StackedAutoencoder(train[0], train[1], train[2], 3)
    ae.create_autoencoder()
    result = ae.evaluate_autoencoder()
    return result[1] * 100
    print("Accuracy: %.2f%%" % (result[1] * 100))


def build_binary_classifiers(path_g1_sg2m, path_g1s_g2m):
    """
    Build the stacked neural network with single output neuron for binary classification to G1 vs. S+G2M and
     G1+S vs. G2M phases, and evaluate its performance
    :param path_g1_sg2m: Path to the labeled dataset in two labels : G1 and SG2M
    :param path_g1s_g2m: Path to the labeled dataset in two labels : G1S and G2M
    :return: Accuracy of classification of each model.
    """
    ############### Ordinal Classifier #################
    dr1 = DatasetReader(path_g1_sg2m)
    dr2 = DatasetReader(path_g1s_g2m)
    binary_train1 = dr1.load_data()
    binary_train2 = dr2.load_data()
    oc1 = OrdinalClassifier(binary_train1[0], binary_train1[1])
    oc2 = OrdinalClassifier(binary_train2[0], binary_train2[1])
    r1 = oc1.classify()
    r2 = oc2.classify()
    return r1, r2

if __name__ == '__main__':
    main()
