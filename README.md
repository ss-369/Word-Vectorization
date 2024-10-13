
---

# Word Vectorization

## Description

This project implements two different methods to train word embeddings:
1. **Singular Value Decomposition (SVD)** applied to a Co-occurrence Matrix.
2. **Word2Vec** using the **Skip-Gram** model with Negative Sampling.

These word embeddings are evaluated using a downstream classification task on the provided **News Classification Dataset**.

## Requirements

- **Language**: Python
- **Framework**: PyTorch
- **Dataset**: News Classification Dataset
  - Use only the **Description** column for training word embeddings.
  - Use the **Label** column for the downstream classification task.

## Features

### 1. Singular Value Decomposition (SVD)
- Build a **Co-occurrence Matrix** and apply **SVD** to extract word vectors.

### 2. Skip-Gram with Negative Sampling
- Implement the **Word2Vec Skip-Gram** model with **Negative Sampling** to train word embeddings.

### 3. Downstream Classification Task
- Evaluate both word embedding methods by training an RNN on the downstream **News Classification** task using the provided dataset.
- Use the same RNN architecture for both SVD and Skip-Gram embeddings to ensure a fair comparison.

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score (Micro, Macro)
- Confusion Matrix

## Hyperparameter Tuning

- Experiment with different **context window sizes** for both SVD and Skip-Gram methods.
- Report performance metrics for at least three different window sizes.
- Analyze which window size performs the best and provide reasoning for your results.

## How to Run

### 1. Singular Value Decomposition (SVD)

To train word vectors using SVD:
```bash
python svd.py
```

To perform classification using the SVD-trained word vectors:
```bash
python svd-classification.py
```

### 2. Skip-Gram with Negative Sampling

To train word vectors using Skip-Gram:
```bash
python skip-gram.py
```

To perform classification using the Skip-Gram-trained word vectors:
```bash
python skip-gram-classification.py
```

### Pretrained Models

1. **SVD Word Vectors**: `svd-word-vectors.pt`
2. **Skip-Gram Word Vectors**: `skip-gram-word-vectors.pt`
3. **SVD Classification Model**: `svd-classification-model.pt`
4. **Skip-Gram Classification Model**: `skip-gram-classification-model.pt`



1. **Source Code**:
   - `svd.py`: Train word embeddings using SVD.
   - `skip-gram.py`: Train word embeddings using Skip-Gram.
   - `svd-classification.py`: Train and evaluate the classification model using SVD word embeddings.
   - `skip-gram-classification.py`: Train and evaluate the classification model using Skip-Gram word embeddings.

2. **Pretrained Models**:
   - `.pt` files containing the pretrained word vectors and classification models.

3. **Report (PDF)**:
   - Hyperparameters used for both SVD and Skip-Gram.
   - Evaluation metrics for the downstream classification task.
   - Analysis and comparison of results.

## Resources
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Word2Vec Explained](http://jalammar.github.io/illustrated-word2vec/)

---

