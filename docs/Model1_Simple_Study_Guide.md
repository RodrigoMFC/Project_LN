# Model 1 (simple): Study Guide

Companion to `Model1_Simple.ipynb`. It explains **what the model does, why, and how to talk about it** in the paper and the oral discussion.

---

## 1. The model in one sentence
Count which words appear in each clinical note (**TF-IDF**), then let a **linear SVM** learn which words point to which of the 12 specialties.

```
note ──► drop 1st keyword ──► join 4 fields ──► TF-IDF vector ──► Linear SVM ──► specialty
```

Lectures used: **T2** (pre-processing, train/test splits), **T5** (documents as vectors, TF-IDF, feature-based ML), **T4 Evaluation** (accuracy, precision, recall, F1).

---

## 2. Step by step

### Step 1: Load the data
`train_clean.csv` (2,669 rows) is the output of `Data_Profile.ipynb`, which repaired broken lines in `train.csv`. Columns: `medical_specialty` (the label), `description`, `sample_name`, `transcription`, `keywords`.

### Step 2: Drop the first keyword
In **every** row with keywords, the first keyword *is* the label (e.g. `cardiovascular / pulmonary, chest pain, …`). That is **label leakage**: the answer is hidden in the input. If we kept it, the model would just read the label instead of learning from the clinical text, and the score would mean nothing. So we remove it, in training **and** in the test file (the same pre-processing must be applied to both, as the brief requires).

### Step 3: Join the four text fields
The model takes one text per row, so we concatenate `description + sample_name + keywords + transcription`. Each field adds information: the transcription is long and detailed, while the description and sample name are short summaries.

### Step 4: Split into train / test (T2)
- The official test file has no labels, so we keep **20%** of the labelled data as **our own test set**. The model never sees it during training.
- `stratify=y`: both parts keep the same class proportions (Dermatology has only 25 rows, so without this the test part could end up with none).
- `random_state=42`: the split is the same every run, so results are **reproducible**.
- Result: 2,135 training rows and 534 test rows.

### Step 5: The model
**(a) `TfidfVectorizer`: text → numbers (T5)**
- Lowercases the text and splits it into words.
- Builds a vector with one dimension per word in the vocabulary (≈13,900 words). Most entries are 0, so it's a **sparse** vector.
- Each value is **TF × IDF**:
  - **TF (term frequency):** how often the word appears in *this* note.
  - **IDF (inverse document frequency):** how rare the word is across *all* notes. Class formula: ln(N / number of notes containing the word).
  - So common words (*the, patient*) get low scores, and distinctive words (*colonoscopy, cataract*) get high ones.
- `sublinear_tf=True`: TF becomes 1 + log(count), so a word repeated 20 times counts more than once, but not 20× more.
- `min_df=2`: ignore words that appear in only one note (mostly typos and names).
- The vectors are normalised to length 1, so long and short notes are comparable (this is what makes **cosine similarity** work, T5).
- We do **not** remove stop words: the brief warns against doing it blindly, and IDF already gives them a low weight.

**(b) `LinearSVC`: the classifier**
- Learns **one weight per word for each specialty**. For a new note it computes, for each specialty, the sum of (word score × weight) and picks the highest.
- "SVM" (Support Vector Machine) means it places the boundary between classes with the largest possible **margin**. That works well with many features and few examples, which is typical for text.
- `class_weight="balanced"`: the data is very **imbalanced** (Surgery 893 rows = 33.5%; Dermatology 25 rows = 0.9%). Without balancing, the model mostly predicts the big classes and small classes get F1 = 0. "Balanced" makes a mistake on a small class cost more (≈36× more for Dermatology than for Surgery).
- `C=0.3`: **regularisation**. A small C gives a simpler model that is less likely to memorise the training data (overfitting). A large C fits the training data more closely.

**Why not Naive Bayes (seen in T5)?** It's the classic algorithm from the course and the official baseline uses it, but on long notes it heavily favours the large classes. A linear SVM is the standard, stronger choice for TF-IDF text classification.

### Step 6: Evaluate (T4)
- **Accuracy** = correct predictions / all predictions.
- **Precision** for class X = of all notes predicted as X, how many really are X? → TP / (TP + FP)
- **Recall** for class X = of all real X notes, how many did we find? → TP / (TP + FN)
- **F1** = 2·P·R / (P + R), the balance between the two.
- **Macro-F1** = the average F1 over the 12 classes (every class counts equally, unlike accuracy).

### Step 7: Look at the mistakes
The brief insists on looking at outputs, not only numbers. The notebook lists the mistakes and checks why they happen (see §4).

### Step 8: Predict the official test file
- `test_no_labels.csv` has **no header row**: line 1 is already a record. It is read with `header=None` and the column names are given manually. Otherwise the first record is lost and every prediction shifts by one line.
- The same steps are applied (drop the first keyword, join the fields).
- The model is **retrained on all 2,669 labelled rows** before predicting (more data → slightly better model).
- The 398 predictions are saved as `results_model1.txt`. The final `results.txt` must come from the **best** model.

---

## 3. Results (own test set, 534 rows)

| Metric | Value | Project gate |
|---|---|---|
| Accuracy | **0.442** | > 0.49 ✗ |
| Worst-class F1 | **0.203** (Radiology) | ≥ 0.25 ✗ |
| Macro-F1 | 0.490 | – |

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Cardiovascular-Pulmonary | 0.455 | 0.403 | 0.427 | 62 |
| Dermatology | 0.750 | 0.600 | 0.667 | 5 |
| Gastroenterology | 0.431 | 0.641 | 0.515 | 39 |
| General Medicine | 0.576 | 0.905 | 0.704 | 42 |
| Neurology | 0.375 | 0.324 | 0.348 | 37 |
| Neurosurgery | 0.250 | 0.438 | 0.318 | 16 |
| Obstetrics-Gynecology | 0.462 | 0.667 | 0.545 | 27 |
| Ophthalmology | 0.483 | 0.933 | 0.636 | 15 |
| Orthopedic | 0.342 | 0.433 | 0.382 | 60 |
| Psychiatry-Psychology | 0.750 | 0.750 | 0.750 | 8 |
| Radiology | 0.229 | 0.182 | 0.203 | 44 |
| Surgery | 0.519 | 0.302 | 0.382 | 179 |

This model does **not** need to pass the gates. Only the submitted `results.txt` (the best model) is scored. Model 1 is the starting point the other models are compared against.

---

## 4. Why the score is limited (main Discussion point)

**The same note appears several times with different specialties.** The data comes from MTSamples (Kaggle), where one note can be filed under several specialties. For example, a knee operation appears both as *Orthopedic* and as *Surgery*.
- 57.9% of the rows share their transcription with a row of another specialty.
- The most common pairs are *X + Surgery* (Cardio, Ortho, Gastro, ObGyn…) and *X + Radiology*.
- So the labels mix two different questions: **which organ system** and **which type of act** (operation or imaging).

**Effect on the model:** if one copy of a note is in the training part as *Orthopedic* and another copy is in the test part as *Surgery*, the model has learned "this note → Orthopedic" and gets the test copy wrong. No model that only reads the text can get both copies right.

**Evidence from the notebook:** **228 of the 298 mistakes (77%)** predict a specialty that the **same note** has elsewhere in the data. Most "mistakes" are therefore valid answers that the single-label evaluation counts as wrong.

**Other limits of word counting:**
- **No word order or negation:** "no evidence of fracture" still contains *fracture*.
- **No synonyms:** *CT* and *computed tomography* are unrelated features.
- **Radiology is the hardest class:** an MRI *of the brain* uses neurology words, so the model predicts Neurology. The model sees the organ, not the type of exam.

These motivate Model 2 (dense vectors that capture meaning) and Model 3 (a contextual Transformer).

---

## 5. Likely oral-exam questions

| Question | Short answer |
|---|---|
| Why remove the first keyword? | It is the label itself (label leakage). Keeping it means the model reads the answer. |
| What is TF-IDF? | Word frequency in the note × rarity across notes. Rare, distinctive words weigh more. |
| Why `class_weight="balanced"`? | The classes are imbalanced (893 vs 25). Without it, small classes are almost never predicted. |
| What does C do? | Regularisation strength. Small C = simpler model, less overfitting. |
| Why stratify the split? | So every class (even the 25-row one) appears in both parts in the same proportion. |
| Why retrain on all data before predicting the test file? | The hyperparameters are fixed, and more training data gives a slightly better model. |
| Why is accuracy only ≈ 0.44? | Identical notes carry different labels, so 77% of the errors predict another valid label of the same note. |
| Accuracy vs macro-F1? | Accuracy is dominated by big classes (Surgery). Macro-F1 treats all 12 equally. |
| Why no stop-word removal? | The brief warns against it, and IDF already down-weights very common words. |

---

## 6. What goes in the paper (template sections)
- **Models:** TF-IDF (sublinear TF, min_df = 2) + LinearSVC (C = 0.3, class_weight = balanced); input = 4 fields joined, first keyword removed. Explain each hyperparameter in one line.
- **Experimental setup:** stratified 80/20 split, random_state = 42; metrics: accuracy, per-class F1, macro-F1.
- **Data analysis observations:** (1) the first keyword equals the label in 100% of rows with keywords; (2) 57.9% of rows share their note with another specialty.
- **Results:** the tables in §3 and the confusion matrix from the notebook.
- **Discussion:** 77% of errors are "same note, other label"; Radiology vs Neurology confusion; examples from the notebook's mistakes table.
- **References:** scikit-learn (TfidfVectorizer, LinearSVC). Declare the use of an LLM (Claude) to help build and explain the model.

---

## 7. Next steps
1. **Data augmentation** (mandatory in the paper), e.g. synonym replacement or back-translation for the small classes (T2). Augment only the training part, never the test part.
2. **Model 2 (intermediate):** dense vectors / embeddings (T7).
3. **Model 3 (advanced):** fine-tuned Transformer (T6, T7, T10). It will be the submitted model, and it must reach the gates.
