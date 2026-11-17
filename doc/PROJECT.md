# Tweet Emotion Detection: A Multiclass Text Classification Study with Classical and Embedding-Based NLP Approaches

## 1. Project Overview

This project builds and compares NLP models that predict the emotion expressed in a tweet. It is the final project of the Natural Language Processing course (Dr. Sotheara Leang, brief dated 28 September 2026). The work is done in a group of five.

The task is multiclass text classification with four labels: anger, joy, optimism, and sadness. Each tweet has exactly one label.

The core focus is on applying the methods taught in class, from problem formulation to evaluation:

- Text preprocessing for noisy social-media text
- Classical text representation (BoW, TF-IDF)
- Word embeddings (pretrained GloVe, Skip-gram trained on the corpus)
- Text classification algorithms (Naive Bayes, Logistic Regression, Linear SVM)
- Evaluation with macro-F1 and per-class metrics
- Comparative and error analysis

The project must apply **at least two** NLP approaches covered in class and compare their performance.

## 2. Project Objectives

1. Formulate emotion detection as a four-class classification problem.
2. Analyze the dataset through question-driven exploratory data analysis (EDA).
3. Build one shared, reproducible preprocessing pipeline and test its choices as experiments.
4. Convert text into numerical features using classical and embedding-based representations.
5. Train classifiers for at least two approaches under the same protocol.
6. Evaluate all models on the same splits using accuracy, macro-F1, per-class metrics, and confusion matrices.
7. Compare approaches and explain the differences with evidence, including error analysis.
8. Deliver a technical report, organized Jupyter notebooks, and the dataset in one ZIP file.

## 3. Course Alignment

| Week | Lesson | Project use |
|---|---|---|
| 1 | Introduction to NLP | Problem formulation, task definition |
| 2 | Text Preprocessing | Shared cleaning pipeline and ablations (Labs 02A, 02B) |
| 3 | Classical Text Representation | BoW and TF-IDF features |
| 4 | Text Classification | Naive Bayes, Logistic Regression, SVM, evaluation metrics (Labs 03A, 04) |
| 5 | Word Embeddings | GloVe and Skip-gram features (Labs 05A, 05B) |
| 6 | Midterm Exam | Not applicable |
| 7 | N-gram Language Models | Optional extension: one language model per class |
| 8 | Zero-Probability N-grams | Optional extension: smoothing for the n-gram classifier |
| 9 | Recurrent Neural Networks | Optional extension: LSTM/GRU classifier |
| 10 | Project Presentation and Final Exam | Defense (15%) |

Grading context: the final project is 30% of the course (report 15%, defense 15%). Using AI tools to complete an entire assignment is prohibited by the course rules, so every member must be able to explain and defend their own work.

## 4. Dataset

### 4.1 Data Source

- Dataset: TweetEval, `emotion` subset (`cardiffnlp/tweet_eval` on HuggingFace)
- Size: about 5.05k tweets

| Column | Description |
|---|---|
| `text` | Textual content of the tweet |
| `label` | Emotion label of the tweet |

### 4.2 Splits and Labels

| Split | Records |
|---|---|
| Train | 3,257 |
| Validation | 374 |
| Test | 1,421 |

Label ids: `0 = anger`, `1 = joy`, `2 = optimism`, `3 = sadness`.

```python
from datasets import load_dataset
ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
```

### 4.3 Dataset Characteristics to Verify in EDA

- The classes are imbalanced. Anger is the largest and optimism the smallest, so macro-F1 is the main metric.
- Mentions are anonymized as `@user`.
- Tweets contain hashtags, emojis, slang, elongated words (for example "sooo"), and URLs.

### 4.4 Data Rules

1. Tune hyperparameters and select models on the **validation** split only.
2. Evaluate on the **test** split once, at the end, for the final results table.
3. Keep a local copy of all three splits in `data/`, because the dataset is part of the submission.

## 5. Experiment Pipeline

### 5.1 Pipeline Architecture

```text
Data Loading (TweetEval emotion)
      |
      v
Exploratory Data Analysis
      |
      v
Text Preprocessing (shared clean() function)
      |
      v
Text Representation (BoW, TF-IDF, GloVe, Skip-gram)
      |
      v
Model Training (NB, Logistic Regression, SVM, MLP)
      |
      v
Evaluation on Validation (tuning)
      |
      v
Final Test Evaluation (once)
      |
      v
Comparison and Error Analysis
      |
      v
Technical Report and Submission
```

### 5.2 Pipeline Steps

| Step | Notebook | Output |
|---|---|---|
| 1. Load and save data | `01_eda.ipynb` | `data/train.csv`, `data/validation.csv`, `data/test.csv` |
| 2. EDA | `01_eda.ipynb` | EDA figures in `results/figures/eda/` |
| 3. Preprocessing | `02_preprocessing.ipynb` | `data/processed/{train,validation,test}.csv` |
| 4. Representation | `03_representation.ipynb` | `data/features/` with `index.csv` |
| 5. Training and evaluation | `04_training_evaluation.ipynb` | Metrics, confusion matrices |
| 6. Results and comparison | `05_results_comparison.ipynb` | Final comparison table, error analysis |

Current status (from `GUIDELINE.md`): notebooks 01, 02, and 03 are done; 04 and 05 are pending.

## 6. NLP Components

### 6.1 Text Preprocessing

Preprocessing is treated as an experiment, not a fixed choice. Every notebook imports one shared `clean()` function from `src/preprocess.py`.

Steps:

- Lowercase the text.
- Remove URLs.
- Replace `@user` with one token or remove it.
- Convert hashtags to words (`#happy` becomes `happy`) because the word carries emotion.
- Convert emojis to words with `emoji.demojize`, since they are strong emotion signals.
- Normalize elongated words (`sooo` becomes `soo`) and contractions.
- Tokenize with `nltk.TweetTokenizer`.

Ablations to run and report:

| Variant | Description |
|---|---|
| Clean text | Cleaning and normalization only |
| No stop words | Clean text with stop words removed, keeping negations |
| No stop words + lemma | As above, with lemmatization |

Removing negations such as "not", "no", and "never" can flip meaning, so negations are kept. The effect of each variant on macro-F1 must be reported.

### 6.2 Text Representation

| Representation | Variants | Course topic |
|---|---|---|
| Bag of Words | Count and binary | Week 3 |
| TF-IDF | Unigram, and unigram + bigram; `min_df` tuned; optional character 2-5 grams | Week 3 |
| Pretrained embeddings | GloVe-Twitter (for example `glove-twitter-100` via gensim), averaged and TF-IDF-weighted | Week 5 |
| Trained embeddings | Skip-gram trained on the training split, averaged | Week 5 |

Feature variants are indexed in `data/features/index.csv`. Vectorizers are fitted on the training split only.

### 6.3 Classification Algorithms

| Algorithm | Used with | Course topic |
|---|---|---|
| Multinomial Naive Bayes | BoW, TF-IDF | Week 4 |
| Logistic Regression | TF-IDF, embeddings | Week 4 |
| Linear SVM | TF-IDF | Week 4 |
| MLP | Embeddings (optional) | Week 5 |

### 6.4 Approaches Compared

| Approach | Representation | Models | Status |
|---|---|---|---|
| 1. Classical | TF-IDF (1-2 gram words, optional character n-grams) | NB, Logistic Regression, Linear SVM | Required |
| 2. Word embeddings | Averaged GloVe-Twitter, compared with self-trained Skip-gram | Logistic Regression, MLP | Required |
| 3. Lexicon and hand-crafted features | Lab 04 "six features" combined with TF-IDF | Logistic Regression | Optional |
| Extensions | n-gram LM per class (Weeks 7-8), RNN (Week 9) | LM likelihood, LSTM/GRU | Optional |

A fine-tuned transformer (for example `cardiffnlp/twitter-roberta-base`) goes beyond the syllabus. Confirm with the lecturer that it counts before using it as a main approach.

### 6.5 Training and Evaluation Protocol

- Use the same splits and the same metrics for every approach.
- Fix `random_state=42` everywhere.
- Tune with `GridSearchCV` and a `PredefinedSplit` over train and validation, or a simple loop over validation.
- Test `class_weight="balanced"` and report its effect, especially on optimism.

Metrics:

- Accuracy
- **Macro-F1** (main metric, the official TweetEval metric)
- Per-class precision, recall, and F1
- Confusion matrix

### 6.6 Error Analysis

Examine 10-20 misclassified tweets and explain the causes, such as sarcasm, confusion between joy and optimism, negation, and short or ambiguous text. Link the findings back to the EDA and preprocessing results.

### 6.7 Final Comparison Table

| Approach | Preprocessing | Representation | Model | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| 1 | | TF-IDF | Logistic Regression | | |
| 2 | | GloVe-Twitter average | Logistic Regression | | |

The table matches the template used in the project progress report.

## 7. Repository Architecture

```text
project-repo/
├── data/                      # train/validation/test CSV, processed/, features/
├── doc/
│   ├── PROJECT.md             # this document
│   └── TASK.md                # team responsibilities
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_representation.ipynb
│   ├── 04_training_evaluation.ipynb
│   └── 05_results_comparison.ipynb
├── src/preprocess.py          # shared clean() function
├── results/                   # metrics JSON/CSV, figures, confusion matrices
├── report/                    # technical report (docx/pdf)
└── requirements.txt
```

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## 8. Technology Stack

| Purpose | Tools |
|---|---|
| Language and environment | Python 3.12, Jupyter, virtual environment |
| Data | HuggingFace `datasets`, pandas, NumPy |
| NLP | NLTK (`TweetTokenizer`, stop words, lemmatizer), `emoji` |
| Representation | scikit-learn vectorizers, gensim (GloVe, Word2Vec Skip-gram) |
| Models | scikit-learn (NB, Logistic Regression, LinearSVC, MLP) |
| Visualization | matplotlib, seaborn |
| Version control | Git |

## 9. Deliverables and Submission

| Item | Requirement |
|---|---|
| Technical report | EDA, preprocessing, representation, classification algorithms, training and evaluation, results, comparative analysis, conclusions. Detailed explanations are graded. |
| Code | Jupyter notebooks covering loading, analysis, preprocessing, representation, training, evaluation, results, comparison |
| Dataset | All splits and processed data used in the project |
| Submission | One ZIP uploaded by the team leader through the eLearning system |
| File name | `Tweet-Emotion-Detection-[GroupNumber]-[TeamNumber].zip` (example: `Tweet-Emotion-Detection-Group01-Team01.zip`) |
| Deadline | To be updated by the lecturer |

Report outline:

1. Introduction
2. Dataset and EDA
3. Preprocessing
4. Text representation
5. Classification algorithms
6. Training and evaluation setup
7. Results
8. Comparative and error analysis
9. Conclusion and future work
10. References

## 10. Project Scope and Limitations

### Included in Scope

- Four-class emotion classification of TweetEval tweets.
- EDA and a shared preprocessing pipeline with ablations.
- BoW, TF-IDF, pretrained GloVe, and Skip-gram representations.
- Naive Bayes, Logistic Regression, and Linear SVM classifiers, with MLP on embeddings.
- Evaluation with accuracy, macro-F1, per-class metrics, and confusion matrices.
- Comparative analysis and error analysis.

### Excluded from Scope (unless confirmed with the lecturer)

- Fine-tuning large pretrained transformers as a main approach.
- Multi-label emotion detection.
- Deployment as a web application.
- Languages other than English.

### Limitations

1. **Small Dataset:** About 5k tweets limits how well trained embeddings and neural models generalize.
2. **Class Imbalance:** Optimism has the fewest examples, which affects its recall and macro-F1.
3. **Noisy Text:** Sarcasm, slang, and emojis are hard for bag-of-words style features.
4. **Averaged Embeddings:** Averaging word vectors discards word order and negation scope.
5. **Single Test Evaluation:** The test split is used once, so the final results come from one run per approach.
