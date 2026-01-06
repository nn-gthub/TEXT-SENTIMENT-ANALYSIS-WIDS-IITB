ASSIGNMENT 2
# Comparative Analysis: Classical vs. Neural NLP

## Project Overview
This project compares two different approaches to Natural Language Processing (NLP) for sentiment analysis: a classical statistical model and a modern neural network. The objective was to determine if the added computational complexity of neural networks yields significant accuracy improvements for fine-grained sentiment classification.

* **Topic:** COVID-19 Tweet Sentiment Analysis
* **Dataset:** COVID-19 NLP Text Classification (Kaggle)

## Methodologies

### System A: Classical Approach (Baseline)
* **Technique:** TF-IDF Vectorization (Bag of Words)
* **Model:** Logistic Regression
* **Logic:** Assigns sentiment based on the frequency of specific words (e.g., "panic", "crisis") without considering word order.

### System B: Neural Approach (Deep Learning)
* **Technique:** Learned Word Embeddings
* **Model:** Keras Sequential Network (Embedding Layer $\rightarrow$ Global Average Pooling $\rightarrow$ Dense Layer)
* **Logic:** Maps words to dense vectors where synonyms are mathematically close. This allows the model to capture semantic meaning and intensity (e.g., understanding that "catastrophe" is stronger than "bad").

## Experimental Results

| Metric | System A (Classical) | System B (Neural) |
| :--- | :--- | :--- |
| **Model Architecture** | TF-IDF + Logistic Regression | Embeddings + Neural Network |
| **Accuracy** | 56.40% | 66.22% |
| **Training Time** | < 5 Seconds | ~30 Seconds |

## Key Observations
1.  **Performance Gap:** The Neural Network outperformed the classical baseline by approximately **10%**.
2.  **Semantic Context:** The classical model struggled with fine-grained labels (like distinguishing "Extremely Positive" from "Positive") because it treats words independently. The neural model successfully captured the *intensity* of words using embeddings.
3.  **Confusion Matrix:** The neural network produced a much cleaner diagonal in the confusion matrix, whereas the classical model showed significant "smearing" between related classes.

## Repository Contents
* week2_asst_notebook.ipynb`: The Python notebook containing the implementation of both System A and System B.
* week2_asst_report_nn.pdf`: A detailed report covering the theoretical logic, architecture diagrams, and error analysis.

## Tech Stack
* **Language:** Python 3
* **Libraries:** TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib
