# Stacked_Autoencoder_Neural_Network_Cell_classification
Classification of RNA biological cells to their cycle stages, using a stacked autoencoder neural network
--------------------------------------------------------------------------------------------------------

Single cell RNA-sequencing concentrates on extracting genetic information from that cell including expression profile, in which millions of reads are generated to indicate whether some gene is expressed in a cell and to what extent.
This data is used in research and as input data to health systems. The evolution of each cell consists of three phases: G1, S, G2.

Data Preperation
----------------
We use the cell read count of 38293 genes in each of the 288 cells (96 samples of each cycle phase).
• We remove all genes with read count of 0 in all cells.
• Remove all genes that are expressed (read count greater than 0) only in less than 20% of the cell samples. Choosing greater percentage of samples for considering the expressiveness of a gene (for example, 60% as was initially intended) can eliminate genes that are related to one specific cell cycle phase, and are most meaningful for classification.
• Normalize the data by applying logarithm on all values of read counts.

Building the Autoencoder neural network
---------------------------------------
We build a network consisting of three fully connected dense hidden layers of 3000, 1000 and 30 neurons, respectively.
After fitting the auto-encoder on the designed model, we froze the weights of the trained layers and replaced the last reconstruction layer.
We measure the accuracy of classificaion.

![Screenshot](Stacked_Autoencoder_Neural_Network_Cell_classification/Img/Autoencoder_scheme_A.png)
![Screenshot](Stacked_Autoencoder_Neural_Network_Cell_classification/Img/Autoencoder_scheme_B.png)

GUI
---------------------------------------
The GUI enables to upload the data files, consisting of cell read counts, and present the evaluation results of the classification.

![Screenshot](Stacked_Autoencoder_Neural_Network_Cell_classification/Img/GUI_screenshot.png)
