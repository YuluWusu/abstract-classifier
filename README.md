**ArXiv Abstract Topic Classifier**


Introduction


This project is an AI application designed to process and categorize academic research abstracts. By analyzing the text from the ArXiv dataset, the system can identify the primary field of study (e.g., Computer Science, Mathematics, Physics) that a paper belongs to. It compares different text vectorization methods to achieve the best classification performance.

Core Features
Data Extraction: Automatically loads and filters large-scale academic abstracts from the ArXiv library.

Text Preprocessing: Cleans raw text by removing special characters, numbers, and normalizing formatting for better AI analysis.

Multiple Vectorization Techniques: Implements Bag of Words (BoW), TF-IDF, and state-of-the-art Sentence Transformers (Embeddings).

Unsupervised Classification: Uses K-Means clustering to group similar research papers and map them to their respective academic categories.

Performance Evaluation: Provides detailed accuracy reports and comparisons between different NLP approaches.

Technologies Used
Python

Scikit-learn (Machine Learning algorithms)

Sentence-Transformers (Deep Learning embeddings)

Hugging Face Datasets (Data sourcing)

NumPy & Matplotlib (Data processing and visualization)

How It Works
The system follows these logical steps:

Data Loading: Fetches abstracts specifically from 'astro-ph', 'cond-mat', 'cs', 'math', and 'physics'.

Vectorization: Converts text into numerical vectors using three different methods to see which captures the most meaning.

Clustering: Applies K-Means to find patterns in the data and assigns labels based on the most common category in each cluster.

Testing: Evaluates the model's ability to predict topics on a separate test set.

Installation
To run this project locally, ensure you have Python installed, then:

Clone the repository: git clone https://github.com/YuluWusu/abstract-classifier.git

Install required libraries: pip install datasets sentence-transformers sklearn matplotlib seaborn numpy

Run the script: python main.py
