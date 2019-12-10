# USU CS5665 Final Project 
Poster Presentation

<object data="poster-HaixuanGuo.pdf" type="application/pdf" width="900px" height="auto">
    <embed src="poster-HaixuanGuo.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/HelenGuohx/ds_projects/blob/master/quora-insincere-questions/poster-HaixuanGuo.pdf">Download PDF</a>.</p>
    </embed>
</object>

### Abstract
Quora is a platform where people can ask questions and others contribute quality answers based on their expertise and experiences. However,
some people may misuse this platform to cause devastation by asking insincere questions that have a non neutral tone, sexual content, are disparaging, inﬂammatory
and not grounded to reality. In this project, a basic text mining procedure is demonstrated including text cleaning, feature selection, tokenizing, vectorization. Some machine learning techniques such as Logistic Regression, SVM are employed to identify insincere questions. For word vectorization, three word embedding methods are compared against each other in two aspects - word coverage and running time.
It shows that GloVe covers most of the words in the dataset while fasttext is the fastest to map words to vectors. SVM performs best in classification in all three models. What's more, PCA is introduced to SVM model but its performance gets worse.  



#### keywords: Text Mining, Word Embedding, SVM, Logistic Regression, F-score

### folder structure

```
.
├── __pycache__
│   └── main.cpython-37.pyc
├── embeddings # embedding files, is empty now
├── embeddings.zip # unzip this file in embeddings folder
├── main.ipynb  # main python code
├── main.py
├── main_export.ipynb  # export result
├── misspell_words.json
├── modeling.py
├── oov_rate.png
├── pca_curve.png
├── rare_words.json
├── result.txt
├── test.py
├── test_output.csv
├── train.csv
├── train_output.csv
└── visualize.ipynb  # visualizing result 



```

### reference
- https://www.kaggle.com/sunnymarkliu/more-text-cleaning-to-increase-word-coverage
- https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora
- https://arxiv.org/ftp/arxiv/papers/1911/1911.01217.pdf
- https://medium.com/@amitbalharakr93/quora-insincere-question-classification-2f19a973273b
- https://pdfs.semanticscholar.org/f469/033055a13eb749808db61c120c12a8cb2bf3.pdf
- https://blog.csdn.net/qq_27802435/article/details/81201357
- https://towardsdatascience.com/quora-insincere-questions-classification-d5a655370c47
 
 
