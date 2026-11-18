# Team Responsibilities and Task Assignment

## Tweet Emotion Detection (NLP Final Project)

## 1. Team Overview

The project is divided into five main responsibilities. Each member owns a major component of the pipeline, with specific tasks and deliverables.

| No. | Assigned Member | Main Responsibility | Primary Focus |
| --- | --- | --- | --- |
| 1 | Sam Sotheara  | EDA | Question-driven EDA |
| 2 | Phon Seaklang | Text Preprocessing | Shared `clean()` pipeline, preprocessing ablations |
| 3 | Ty Punleu | Text Representation | BoW, TF-IDF, GloVe, Skip-gram features |
| 4 | Song Kimleangchhay | Model Training & Tuning | Classical and embedding-based classifiers, hyperparameter tuning |
| 5 | Srey Sitharath | Evaluation, Comparison & Error Analysis | Metrics, final test run, comparison table, error analysis |

Each member is responsible for implementing and documenting their assigned component. The project is also a shared learning exercise: all members must understand the complete pipeline and be able to explain and defend their own work and the whole system during the defense.

---

## 2. Individual Responsibilities and Tasks

### 1. Sam Sotheara : EDA

**Main responsibility:** Understand the dataset and be able to inteprete the dataset description and EDA process.
#### Subtasks
**2.1. Exploratory Data Analysis (`01_eda.ipynb`)**

- Analyze class distribution and imbalance.
- Analyze tweet length, vocabulary size, and word frequency.
- Show top and characteristic words per emotion.
- Measure Twitter elements: hashtags, emojis, `@user`, URLs, elongated words.
- Study negation and other cues that affect emotion.
- Save figures to `results/figures/eda/`.

#### Deliverables

1. `01_eda.ipynb`
2. Report sections: Dataset and EDA.

**Expected handover:** A documented of dataset and EDA process.
---

### 2. Phon Seaklang:  Text Preprocessing

**Main responsibility:** Build the shared preprocessing pipeline used by every approach.

#### Subtasks
**2.1. Preprocessing Pipeline (`02_preprocessing.ipynb`)**

- Lowercase, remove URLs, and handle `@user`.
- Convert hashtags to words and emojis to words.
- Normalize elongation and contractions.
- Tokenize with `nltk.TweetTokenizer`.
- Write the final logic into `src/preprocess.py` as one `clean()` function.

**2.2. Preprocessing Ablations**

- Compare clean text, no stop words, and no stop words with lemmatization.
- Keep negations when removing stop words.
- Report how each variant changes vocabulary size and, later, macro-F1.

**2.3. Output**

- Save processed splits to `data/processed/{train,validation,test}.csv`.
- Document the reason for every preprocessing step.

#### Deliverables

1. `02_preprocessing.ipynb`.
2. `src/preprocess.py` with the shared `clean()` function.
3. Processed datasets and preprocessing figures.
4. Report sections:  Preprocessing.

**Expected handover:** A documented, reusable preprocessing function and processed splits that all approaches use.

---

### 3. Ty Punleu: Text Representation

**Main responsibility:** Convert preprocessed text into the numerical features used by the classifiers.

#### Subtasks

**3.1. Classical Representations (`03_representation.ipynb`)**

- Build BoW features (count and binary).
- Build TF-IDF features (unigram, and unigram + bigram).
- Optionally add character 2-5 gram TF-IDF.
- Tune `min_df` and report its effect.
- Fit vectorizers on the training split only.

**3.2. Pretrained Embeddings**

- Load a Twitter-trained GloVe model (for example `glove-twitter-100` via gensim).
- Build averaged document vectors.
- Build TF-IDF-weighted document vectors.

**3.3. Trained Embeddings**

- Train a Skip-gram model on the training text (Lab 05B).
- Build averaged document vectors from it.
- Compare coverage and quality against pretrained GloVe.

**3.4. Feature Storage**

- Save features and vectorizers to `data/features/`.
- Maintain `index.csv` listing every representation and preprocessing variant.
- Visualize representations (top TF-IDF words, 2D projection, centroid similarity).

#### Deliverables

1. `03_representation.ipynb`.
2. Saved feature matrices, vectorizers, and the Skip-gram model.
3. Feature index (`data/features/index.csv`) and representation figures.
4. Report section: Text Representation (how BoW, TF-IDF, and embeddings work, and why each was chosen).

**Expected handover:** A set of ready-to-use feature matrices for train, validation, and test, with no information leaked from the test split.

---

### 4. Song Kimleangchhay: Model Training & Tuning

**Main responsibility:** Train and tune the classifiers for each approach under one common protocol.

#### Subtasks

**4.1. Approach 1: Classical (`04_training_evaluation.ipynb`)**

- Train Multinomial Naive Bayes, Logistic Regression, and Linear SVM on TF-IDF features.
- Compare BoW and TF-IDF inputs.
- Establish the quick TF-IDF + Logistic Regression baseline on validation.

**4.2. Approach 2: Word Embeddings**

- Train Logistic Regression (and optionally an MLP) on averaged GloVe vectors.
- Train the same models on Skip-gram vectors.
- Compare averaged and TF-IDF-weighted embeddings.

**4.3. Optional Approach 3**

- Implement the lexicon and hand-crafted features from Lab 04 (positive and negative word counts, `!`, emojis, pronouns).
- Combine them with TF-IDF.
- Optional extensions from Weeks 7-9: n-gram LM per class with smoothing, or an LSTM/GRU classifier. Agree on these with the team first.

**4.4. Tuning and Imbalance**

- Tune with `GridSearchCV` and `PredefinedSplit`, or a loop over validation.
- Test `class_weight="balanced"` and report the effect on optimism.
- Keep `random_state=42` and record the chosen hyperparameters.

#### Deliverables

1. Training code in `04_training_evaluation.ipynb`.
2. Trained models and tuning results saved in `results/`.
3. Validation scores for every approach and configuration.
4. Report sections: Classification Algorithms (short theory for NB, Logistic Regression, SVM), Training Setup.

**Expected handover:** The best configuration per approach, selected on validation only, ready for the final test run.

---

### 5. Srey Sitharath: Evaluation, Comparison & Error Analysis

**Main responsibility:** Evaluate all approaches consistently, run the final test evaluation once, and explain the results.

#### Subtasks

**5.1. Evaluation Framework**

- Compute accuracy, macro-F1, and per-class precision, recall, and F1 using one shared function.
- Produce confusion matrices as PNG files in `results/`.
- Save metrics as JSON or CSV.

**5.2. Final Test Evaluation (`05_results_comparison.ipynb`)**

- Take the best validation configuration per approach from Member 4.
- Evaluate each on the test split exactly once.
- Build the final comparison table (approach, preprocessing, representation, model, accuracy, macro-F1).

**5.3. Comparative Analysis**

- Explain why one approach performs better, using EDA and ablation evidence.
- Discuss the effect of preprocessing, class weighting, and representation.
- Discuss trade-offs: accuracy, speed, interpretability.

**5.4. Error Analysis**

- Review 10-20 misclassified tweets.
- Categorize causes: sarcasm, joy versus optimism confusion, negation, short text, emoji-only text.
- Link each cause to a possible improvement.

#### Deliverables

1. `05_results_comparison.ipynb`.
2. Final results table and confusion matrices.
3. Error analysis with example tweets.
4. Report sections: Results, Comparative and Error Analysis.

**Expected handover:** A complete, evidence-based comparison that answers which approach works best and why.

---

## 3. Shared Responsibilities for All Members

The purpose of the project is to learn how an NLP classification system works, not to complete five independent coding tasks.

### 3.1. Project Setup and Flow discussion
**Main responsibility:** Set up the repository and data, keep the pipeline consistent across notebooks, write the framing sections of the report, and submit the final ZIP.

#### Subtasks

**1.1. Repository and Environment**

- Create the folder layout and `requirements.txt`.
- Set up the virtual environment and push the repository to GitHub.
- Keep notebooks runnable top-to-bottom.

**1.2. Data Loading**

- Load `cardiffnlp/tweet_eval` (emotion) with HuggingFace `datasets`.
- Save train, validation, and test CSV files to `data/`.
- Confirm split sizes: 3,257 train, 374 validation, 1,421 test.

**1.3. Integration and Review**

- Review pull requests and notebook outputs from every member.
- Confirm all notebooks import the shared `clean()` function.
- Check that the test split is never used for tuning.

**1.4. Report Framing**

- Write the Introduction (problem, goal, course alignment).
- Write the Conclusion and Future Work.
- Merge all sections into one consistent report.

**1.5. Submission**

- Re-run all notebooks top-to-bottom before submission.
- Package the report, notebooks, and dataset into one ZIP:
  `Tweet-Emotion-Detection-[GroupNumber]-[TeamNumber].zip`
- Upload through the eLearning system and keep the progress report updated.

#### Deliverables

1. Repository with agreed structure, `requirements.txt`, and setup instructions.
2. Saved raw dataset splits in `data/`.
3. Report sections: Introduction, Conclusion and Future Work.
4. Integrated, reproducible notebooks.
5. Final submission ZIP.

**Expected handover:** A clean repository where every notebook runs in order and the dataset is included.


### 3.2. Understand the Complete Pipeline

```text
Data Loading
      |
      v
EDA
      |
      v
Preprocessing
      |
      v
Text Representation
      |
      v
Model Training and Tuning
      |
      v
Evaluation
      |
      v
Comparison and Error Analysis
      |
      v
Report and Submission
```

Each member should be able to explain how their component connects to the previous and next stages.

### 3.3. Course Concept Review

Each member should be able to explain, from the lectures and labs, the concepts behind their component and the neighboring ones: preprocessing (Week 2), BoW and TF-IDF (Week 3), classification and metrics (Week 4), and word embeddings (Week 5).

### 3.4. Shared Error Review

All members should review misclassified tweets together and agree on the error categories used in the report.

### 3.5. Integration and Code Review

- Review each other's notebooks before merging.
- Run all notebooks top-to-bottom from a clean environment.

### 3.6. Progress Reports and Defense

- Fill in the team progress report (members, progress percentage, approaches, best results, problems, next steps).
- Prepare the defense together. Each member presents and justifies their own work and answers questions about the whole project.
- Do not use AI tools to complete an entire assignment, as required by the course rules.

---

## 4. Suggested Timeline

| Week | Goal |
|---|---|
| 1 | Repository and data saved, EDA done, preprocessing function agreed |
| 2 | Approach 1 baseline with validation scores, progress report filled |
| 3 | Approach 2, preprocessing ablations, class-weight experiments |
| 4 | Final test-set run, comparison table, error analysis, figures |
| 5 | Report written and reviewed, notebooks re-run, ZIP submitted, defense prepared |

Status from `GUIDELINE.md`: notebooks 01, 02, and 03 are done. Remaining: notebook 04, notebook 05, the report, and the submission.

---

## 5. Final Working Principles

1. **Ownership:** Each member completes and documents their assigned component.
2. **Collaboration:** Coordinate when components depend on each other (preprocessing to representation to training to evaluation).
3. **Shared Learning:** Everyone understands the fundamental concepts, not just their own part.
4. **Reproducibility:** Fixed seeds, shared preprocessing, and notebooks that run from top to bottom.
5. **No Test Leakage:** Tune on validation only. Evaluate on test once at the end.
6. **Evidence-Based Evaluation:** Use macro-F1, per-class metrics, and error analysis instead of a few successful examples.
7. **Same Protocol:** Every approach uses the same splits, metrics, and tuning procedure so the comparison is fair.
8. **Explained Results:** The report must explain each step and each result, not only show numbers. Completeness and clarity are graded.
9. **Integration:** All components work together as one complete project, submitted as a single ZIP.
