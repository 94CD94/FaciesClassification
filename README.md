# FaciesClassification

The idea behind this work is to predict the facies distribution from seismic data even before setting an exploration well. 
This is achieved by exploiting the ML's ability to find a relationship between a set of parameters (Elastic parameters) and one output (Facies). Elastic <b>parameters</b> used for the algorithm's training are gathered from 6 wells of a given study area, and to each set of parameter a facies is assigned. Once the model will be trained and tested, it will be used to predict the facies using the elastic parameters from an AVO inversion (http://dx.doi.org/10.1190/1.1543206) of a 2D seismic line. The best ML algorithm among Support vector machine, Logistic regression, Random Forest, Neural Networks, and K-Nearest neighbors, is selected using a k-fold-cv to find also the best parameters for each algorith. 
The best algorithm based on balanced accuracy score(K- Nearest),  will be able to detect the reservoir's facies from the 2D Seismic line.

 
