# Simple Document Classification using Multi Class Logistic Regression & SVM Soft Margin from scratch
Enseirb-Matmeca, Bordeaux INP | [Anas AHOUZI](https://www.linkedin.com/in/anas-ahouzi-6aab0b155/)
***

## Description
This mini-project contains an implementation from scratch of some Multi-Class Classification Algorithms. The data is already cleaned, and doesn't need any further pre-processing, it was encoded using Tf-idf (Term Frequency Inverse Document Frequency).<br />
**Firstly**, I implemented the Logistic Regression algorithm with One-vs-All strategy to adapt the algorithm for the multi-classification task. I used the Momentum with SGD optimizer for optimizing the Binary Cross-Entropy loss used to get the optimal weight matrix.<br />
**Secondly**, I implemented Multi-Class SVM with the same strategy as Logistic Regression, and since I chose soft margin SVM to deal with non-linearly separable data, I used the same previous optimizer to optimize the L2 reguralized Hinge loss. <br />

## Repository Structure
The repository contains the following files & directories:
- **main.ipynb:** A Jupyter notebook where the main functions of the project are called, and where results are displayed.
- **Loss directory:** It contains an implementation of the various loss functions mentioned in the description, and their corresponding gradient calculus.
- **ML_Algos directory:** This directory contains an implementation of the various ML algorithms mentioned in the description.
- **The dataset:** This mini-project was taken from a HackerRank challenge, you can refer to [the following link](https://www.hackerrank.com/challenges/document-classification/problem), to get the dataset as well as the instructions to solve the problem.

## Next steps
Implementation of SVM with the Kernel trick, to deal with non-linearly separable data using various Kernel functions (Gaussian, Polynomial, etc..).



---
For any information, feedback or questions, please [contact me][anas-email]










[anas-email]: mailto:ahouzi2000@hotmail.fr