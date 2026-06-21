# Decoding Tweets | A Complete NLP Guide (DNN, LSTM)

## Overview
This project serves as a comprehensive, end-to-end blueprint for Natural Language Processing (NLP). Originally developed as a practical guide while learning and mastering the fundamentals of NLP, the architecture built here provides a robust, universal foundation. The core pipelines, data engineering, and comparative methodologies demonstrated in this notebook can be seamlessly adapted to any broad NLP task, ranging from text classification to machine translation.

## The Complete NLP Blueprint & Walkthrough
This notebook systematically details a real-world sentiment analysis project, transitioning from raw Twitter text all the way to a functional, predictive model. The unique value of this project lies in implementing and comparing distinct NLP approaches side-by-side, explicitly bridging the gap between theoretical concepts and code implementation.

The project is structured into six foundational steps:

### 1. Data Loading & Exploratory Data Analysis (EDA)
Understanding the raw data is the critical first step before any transformation occurs.
*   **Implementation:** The data is loaded using `pandas` (handling both training and validation CSV files) and cleaned of any null text entries. 
*   **Visual Insights:** 
    *   Utilized `seaborn` to plot the distribution of target labels, ensuring an understanding of class balance[span_3].
    *   Implemented `WordCloud` to generate visual representations of the most frequent vocabulary across the dataset, offering an intuitive grasp of the corpus.

### 2. Advanced Preprocessing: The Divergence
Text cleaning is not a one-size-fits-all process. The preprocessing strategy must match the model's architecture. This project explicitly compares two distinct text-cleaning pipelines:

*   **Pipeline A: Deep Linguistic Cleaning via `spaCy` (For DNN)**
    *   **Logic:** Standard neural networks rely on statistical feature extraction, meaning noise (like punctuation) and word variations can dilute the signal. 
    *   **Code Connection:** We utilized the `en_core_web_sm` model in `spaCy` to process the text. The custom `preprocessing_pipe` iterates over the tokens to strictly remove stop words (`!token.is_stop`) and punctuation (`!token.is_punct`), while applying lemmatization (`token.lemma_`) to reduce words to their root forms.
    *   **Optimization:** Used `tokenizer.pipe` with `n_process=4` and `batch_size=64` for efficient multi-core processing.

*   **Pipeline B: Sequence Preservation via `Keras Tokenizer` (For RNN/LSTM)**
    *   **Logic:** Recurrent networks require the original sequential order of words to understand context. Stripping too many words destroys the sequence.
    *   **Code Connection:** We bypassed the `spaCy` cleaning and directly used the TensorFlow Keras `Tokenizer`. It converts the raw text into integer sequences (`texts_to_sequences`) and handles unseen words smoothly using the `oov_token="<OOV>"` parameter. The sequences are then unified in length using `pad_sequences` up to the `maxseq_len`.

### 3. Vectorization Strategies: Statistical vs. Contextual
Machine learning models require numerical input. The project compares two fundamental methodologies for converting text to numbers:

*   **TF-IDF Vectorization (Used with DNN):**
    *   **Approach:** An algorithmic approach that scores word importance based on frequency across the document versus the entire corpus. It creates a massive, sparse matrix.
    *   **Code Connection:** Implemented via `sklearn.feature_extraction.text.TfidfVectorizer`. The training data is fitted and transformed, while the validation/test data is strictly transformed to prevent data leakage.

*   **Word Embeddings (Used with RNN/LSTM):**
    *   **Approach:** A deep learning technique where words are mapped into a dense vector space, allowing the model to learn semantic relationships (words with similar meanings cluster together).
    *   **Code Connection:** Implemented as the first layer in the Keras `Sequential` models using the `Embedding` layer, mapping the total `wordscount` to a dense 200-dimensional vector space (`output_dim=200`).

### 4. Model Architectures & Training
Three distinct neural network architectures were constructed from scratch, utilizing the `Sparse Categorical Crossentropy` loss function (enabled by `LabelEncoder` for our target variables) and the `Adam` optimizer:

*   **1. Deep Neural Network (DNN):** 
    *   Built with standard fully connected layers (`Dense(64)` -> `Dense(32)`). It excels at finding broad patterns in the TF-IDF statistical matrix.
*   **2. Simple Recurrent Neural Network (SimpleRNN):** 
    *   Built with an `Embedding` layer followed by a `SimpleRNN(256)` layer. Designed to process the padded integer sequences step-by-step.
*   **3. Long Short-Term Memory (LSTM):** 
    *   Built with an `Embedding` layer followed by an `LSTM(256)` layer. It specifically tackles the vanishing gradient problem of the SimpleRNN by using internal memory gates to retain long-term context.
*   **Training Optimization:** The sequence models utilized an `EarlyStopping` callback (`monitor='val_loss', patience=10`) to automatically halt training when performance plateaued, restoring the best weights to prevent overfitting.

### 5. Head-to-Head Evaluation & Conclusion
*   **Methodology:** All models were rigorously tested against the validation/test set. The evaluation heavily relied on `sklearn.metrics.classification_report` (analyzing precision, recall, and f1-scores) and `confusion_matrix` visualized via `seaborn` heatmaps.
*   **The Winner:** By explicitly comparing the accuracy and loss metrics across the architectures, the **DNN Model** emerged as the superior model for this specific dataset and text classification task.

### 6. Live Pipeline Demonstration
The project lifecycle is completed by applying the winning model to completely unseen, raw sentences. 
*   **The Workflow:** Custom strings (e.g., "Hornet is stronger than Hollowknight!") are passed sequentially through the exact logic used in training: `preprocessing_pipe` (spaCy cleaning) -> `vect.transform` (TF-IDF vectorization) -> `model.predict` (DNN prediction).
*   Finally, `LabelEncoder.inverse_transform` is utilized to map the numerical predictions back to readable text labels (Positive, Negative, Neutral, Irrelevant).
