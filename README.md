# 20 Newsgroups Text Classification and Topic Discovery Study

## Description
This project evaluates and compares two different approaches for text classification using the 20 Newsgroups dataset: high-dimensional TF-IDF features and low-dimensional LDA topic distributions. The study involves preprocessing, unsupervised topic modeling, and supervised classification using Naive Bayes, Logistic Regression, and SVM.

## Project Structure
```
20_newsgroups_study/
├── data/            # Dataset storage (if applicable)
├── notebooks/       # Jupyter notebooks for exploration
├── src/             # Source code (topic_discovery.py)
├── output/          # Generated plots and results
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

## Installation
To set up the environment, ensure you have Python installed and run:
```bash
pip install -r requirements.txt
```

## Usage
To run the full analysis, including data loading, topic discovery, and model evaluation, execute the script from the root directory:
```bash
python src/topic_discovery.py
```

## Results Summary
The following table summarizes the classification accuracy achieved by different models using TF-IDF vs. LDA features:

| Feature   | Model               |   Accuracy |
|:----------|:--------------------|-----------:|
| TF-IDF    | Naive Bayes         |  0.681625  |
| TF-IDF    | Logistic Regression |  0.68906   |
| TF-IDF    | SVM                 |  0.669278  |
| LDA       | Naive Bayes         |  0.0796601 |
| LDA       | Logistic Regression |  0.153346  |
| LDA       | SVM                 |  0.138609  |

### Key Findings
- **TF-IDF vs. LDA:** TF-IDF features significantly outperformed LDA topic distributions across all classifiers. This suggests that the specific keyword signals preserved in TF-IDF are more discriminative for this dataset than the latent semantic topics discovered by LDA.
- **Top Model:** Logistic Regression using TF-IDF features was the best performing configuration.
- **LDA Performance:** The lower accuracy in LDA-based models likely stems from the high degree of information compression when reducing thousands of terms into just 20 topics.
