# Cloud Segmentation

Codes for cloud segmentation in ground-based infrared sky images adquiared using an sky imager mounted on a solar tracker. The codes were run in a High Performances Computer and the library for the paralellization of the code is MPI.

## Generative Models

The generative models include in this repository are: 

Gaussian Mixture Model GMM_segm.py

K-means Clustering KMS_segm.py

Gaussian Discriminat Analysis (Linear) GDA_segm.py

Naive Bayes Classifier NBC_segm.py

## Discriminative Models

The discriminative model were implemented in their primal formulation. The features were transformed to a feature space using split basis functions. See .

Ridge Regression for Classification RRC_segm.py
Suport Vector Machine SVC_segm.py
Gaussian Process for Classification cross-validated in parallel using  GPC-MPI_segm.py

## Markov Random Fields

Unsupervised Gaussian MRF optimized via Independet Conditional Models cross-validated in parallel using MPI ICM-MRF-MPI_segm.py

Supervised Gaussian MRF MRF_segm.py

Supervised Gaussian MRF with Simulate Anneling on the implementation SA-MRF_segm.py

Unsupervised Gaussian MRF optimized via Independet Conditional Models with Simulate Anneling on the implementation cross-validated in parallel using MPI SA-ICM-MRF-MPI_segm.py


## Utils for Data Processing 

The utils for loading, organized the vectors and matrices, processing the data, and common dependecies are in the files utils.py and 
