# Cloud Segmentation

Codes for cloud segmentation in ground-based infrared sky images adquiared using an sky imager mounted on a solar tracker. The codes were run in a High Performances Computer and the library for the paralellization of the code is MPI. See XX.

## Generative Models

The generative models include in this repository are: 

Gaussian Mixture Model (clustering) is GMM_segm.py.

K-means Clustering is KMS_segm.py.

Gaussian Discriminat Analysis (Linear) is GDA_segm.py.

Naive Bayes Classifier is NBC_segm.py.

## Discriminative Models

The discriminative model were implemented in their primal formulation. The features were transformed to a feature space using split basis functions. See XX.

Ridge Regression for Classification is RRC_segm.py.

Suport Vector Machine is SVC_segm.py.

Gaussian Process for Classification cross-validated in parallel using is GPC-MPI_segm.py.

## Markov Random Fields

The MRF implemented in this repository are:

Supervised Gaussian MRF is MRF_segm.py.

Unsupervised Gaussian MRF optimized via Independet Conditional Models cross-validated in parallel using MPI is ICM-MRF-MPI_segm.py.

Supervised Gaussian MRF with Simulate Anneling on the implementation is SA-MRF_segm.py.

Unsupervised Gaussian MRF optimized via Independet Conditional Models with Simulate Anneling on the implementation cross-validated in parallel using MPI is SA-ICM-MRF-MPI_segm.py.

See XX for further information.

## Utils for Data Processing 

The utils for loading, organized the vectors and matrices, processing the data, and common dependecies are in the files utils.py and feature_extraction_utils.py.

## Dataset

A sample dataset is publicaly available in DRYAD repository: https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.zcrjdfn9m
