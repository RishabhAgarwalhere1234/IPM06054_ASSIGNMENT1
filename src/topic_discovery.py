import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

def run_analysis():
    # 1. Load Dataset
    print("Loading 20 Newsgroups dataset...")
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
    y_train, y_test = newsgroups_train.target, newsgroups_test.target

    # 2. Preprocess Text Data (TF-IDF)
    print("Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)
    X_test_tfidf = vectorizer.transform(newsgroups_test.data)

    # 3. Topic Discovery (LDA)
    print("Applying LDA for topic discovery...")
    lda = LatentDirichletAllocation(n_components=20, random_state=42)
    X_train_lda = lda.fit_transform(X_train_tfidf)
    X_test_lda = lda.transform(X_test_tfidf)

    # 4. Train and Evaluate Classifiers
    print("Training and evaluating models...")
    results = []

    # Define model configurations
    configs = [
        ('TF-IDF', 'Naive Bayes', MultinomialNB(), X_train_tfidf, X_test_tfidf),
        ('TF-IDF', 'Logistic Regression', LogisticRegression(solver='liblinear', random_state=42), X_train_tfidf, X_test_tfidf),
        ('TF-IDF', 'SVM', SVC(kernel='linear', random_state=42), X_train_tfidf, X_test_tfidf),
        ('LDA', 'Naive Bayes', GaussianNB(), X_train_lda, X_test_lda),
        ('LDA', 'Logistic Regression', LogisticRegression(solver='liblinear', random_state=42), X_train_lda, X_test_lda),
        ('LDA', 'SVM', SVC(kernel='linear', random_state=42), X_train_lda, X_test_lda)
    ]

    for feat_name, model_name, clf, x_train, x_test in configs:
        clf.fit(x_train, y_train)
        acc = accuracy_score(y_test, clf.predict(x_test))
        results.append({'Feature': feat_name, 'Model': model_name, 'Accuracy': acc})

    # 5. Display and Save Results
    df_results = pd.DataFrame(results)
    print("
Classification Accuracy Summary:")
    print(df_results)
    
    # 6. Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x='Model', y='Accuracy', hue='Feature')
    plt.title('Performance Comparison: TF-IDF vs LDA Topics')
    plt.ylim(0, 1.0)
    
    # Ensure output directory exists and save plot
    os.makedirs('../output', exist_ok=True)
    plt.savefig('../output/performance_comparison.png')
    print("
Chart saved to output/performance_comparison.png")

if __name__ == '__main__':
    run_analysis()