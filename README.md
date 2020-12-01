# Simple Document Classification using Multi Class Logistic Regression & SVM Soft Margin from scratch
## Description
This mini-project contains an implementation from scratch of some Multi-Class Document Classification Algorithms. The data is already cleaned, and doesn't need any further pre-processing, it was encoded using Tf-idf.<br />
**Firstly**, I implemented the Logistic Regression algorithm with One-vs-All strategy to adapt the algorithm for the multi-classification task. I used the Momentum with SGD optimizer for optimizing the Binary Cross-Entropy loss used to get the optimal weight matrix.<br />
**Secondly**, I implemented Multi-Class SVM with the same strategy as Logistic Regression, and since it's soft margin SVM, I used the same previous optimizer to optimize the L2 reguralized Hinge loss. The repository contains the following files & directories: <br />

- **main.ipynb:** A Jupyter notebook where the main functions of the project are called, and where results are displayed.
- **Loss directory:** It contains an implementation of the various loss functions mentioned above, and their corresponding gradient calculus.
- **ML_Algos:** This directory contains an implementation of the various ML algorithms mentioned above.