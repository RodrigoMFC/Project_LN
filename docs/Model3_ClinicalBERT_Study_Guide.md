# Model 3 (advanced): Fine-tuned Clinical BERT, Study Guide

Companion to `Model3_ClinicalBERT.ipynb`. It covers what the model does, why each choice was made, how to run it, and how to talk about it in the paper and the oral discussion.

---

## 1. The system in one sentence
If the note's first keyword names a specialty, answer it (**keyword rule**). Otherwise, **Bio_ClinicalBERT** (a Transformer pre-trained on ~2 million clinical notes, fine-tuned on ours) reads the text and picks the best specialty the note doesn't already have in training (**duplicate-note rule**).

```
note ─► first keyword names a specialty? ── yes ─► that specialty                (320 of 398 test notes)
                    │ no
                    ▼
     drop 1st keyword ─► "sample_name [SEP] description [SEP] keywords [SEP] transcription"
     ─► WordPiece tokens (max 512) ─► 12 Transformer layers ─► [CLS] vector (768)
     ─► linear layer ─► 12 scores ─► duplicate-note rule ─► specialty            (78 of 398 test notes)
```

### 1.1 Why the keyword rule (group decision)
- **Fact:** in all 2,219 training notes with keywords, the first keyword is the label. 320 of the 398 test notes have such a keyword.
- **Justification:** the brief asks to map *"each clinical text (and other given elements related with the clinical text)"*. The keywords are a given element, present in the test file.
- **Honesty:** it's label leakage, not language understanding. The paper must say so and report the score **with and without** it, plus BERT's score on notes **without keywords** (the part BERT really decides). The notebook prints all of these (section 8.3).
- BERT is still trained with the first keyword removed, so it learns from clinical content. That matches its real job: notes with no keywords.

Lectures: **T6** (Transformers, self-attention), **T7** (BERT, pre-training, fine-tuning, [CLS], transfer learning), **T4** (subword tokenization), **T2** (no pre-processing for Transformers), **T10** (fine-tuning cost, PEFT as future work).

---

## 2. How to run it
1. Install (once). The RTX 50-series needs a CUDA 12.8+ build of PyTorch:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install transformers scikit-learn pandas matplotlib
   ```
2. Open `Model3_ClinicalBERT.ipynb` in the project folder and **Run All**.
3. The first cell should print `device: cuda | NVIDIA GeForce RTX 5080`. If it says `cpu`, PyTorch can't see the GPU, so reinstall it with the command above.
4. The first run downloads the model (~400 MB). Training runs twice: once on 80% for evaluation, once on 100% for `results.txt`. Expect **roughly 5–10 minutes** in total on a 5080.
5. Output: **`results.txt`** (398 lines, Unix line endings). It overwrites the file Model 2 wrote, which is intended, since Model 3 is the model you submit.

*The code was tested end to end with a small stand-in model. The real scores only appear when you run it with Bio_ClinicalBERT on your GPU.*

---

## 3. Key concepts

### 3.1 Pre-training vs fine-tuning (T7)
- **Pre-training** (done by others, expensive): BERT learns language by predicting masked words (*Masked Language Modelling*) on huge text collections. Bio_ClinicalBERT = BERT → further trained on biomedical papers (BioBERT) → further trained on clinical notes from MIMIC-III (Alsentzer et al., 2019). It already "knows" that *cholecystectomy* relates to the gallbladder.
- **Fine-tuning** (what we do, cheap): continue training all the weights on *our* task, with a small learning rate, so the general knowledge is adapted, not erased. This is **transfer learning**.

### 3.2 Why a Transformer can do better than TF-IDF (T6)
- **Self-attention:** every token's vector is updated by looking at every other token in the note, so meaning depends on context. "Fracture" in *"no fracture seen"* and in *"open fracture of the tibia"* get different vectors. TF-IDF treats them identically.
- **Synonyms:** pre-training places related terms close together (*CT* ≈ *computed tomography*), so the model generalises beyond the exact words seen in our 2,000 training notes.
- **Word order matters:** TF-IDF is a bag of words.

### 3.3 Subword tokenization (T4)
BERT uses **WordPiece** (similar to BPE). Common words stay whole, and rare ones are split into known pieces, e.g. `cholecystectomy → cho ##le ##cy ##st ##ectomy` (the exact split depends on the vocabulary). There's no unknown-word problem. Every input starts with **[CLS]** and fields are separated by **[SEP]**.

### 3.4 The [CLS] token and the classification head (T7)
After 12 layers, the vector of the [CLS] token (768 numbers) acts as a summary of the whole input. A new **linear layer** (768 → 12) turns it into one score (logit) per specialty. That layer starts random, which is why `transformers` prints *"newly initialized"*. That's expected.

### 3.5 The 512-token limit
BERT can't read more than 512 tokens, and many notes are longer (the notebook prints the exact share). Our choice: put the **short, dense fields first** (sample name, description, keywords), so that truncation only cuts the **end of the transcription**, usually the least informative part (closing, follow-up, sutures). Alternatives for Future Work: split long notes into chunks and average their predictions, or use a long-context model.

### 3.6 No pre-processing (T2)
No lowercasing, stemming or stop-word removal: the model was pre-trained on raw text with its own tokenizer, and "cleaning" makes our text look *less* like what it learned from. (Bio_ClinicalBERT is *cased*, so capitals carry information.) We only remove the leaked first keyword.

---

## 4. Training choices (paper: Experimental Setup → Hyperparameters)

| Setting | Value | Why |
|---|---|---|
| Base model | `emilyalsentzer/Bio_ClinicalBERT` | domain match: pre-trained on clinical notes, like ours |
| Max length | 512 tokens | BERT's maximum |
| Epochs | 4 | BERT authors recommend 2–4 for fine-tuning. More risks overfitting on 2k notes |
| Learning rate | 2e-5 | standard for BERT fine-tuning. Small so pre-trained knowledge isn't destroyed |
| Optimiser | AdamW, weight decay 0.01 | standard for Transformers. Weight decay = mild regularisation |
| Schedule | 10% linear warm-up, then linear decay to 0 | avoids large, destabilising updates at the start, when the new layer is random |
| Batch | 8 × 2 accumulation = effective 16 | fits in 16 GB VRAM at 512 tokens |
| Loss | cross-entropy with balanced class weights | same idea as `class_weight="balanced"` in Model 1: rare classes count more |
| Gradient clipping | max norm 1.0 | prevents occasional huge updates |
| Precision | bf16 mixed precision | ~2× faster and less memory on RTX 50-series, with no meaningful accuracy loss |
| Seed | 42 everywhere | reproducible split, initialisation and batch order |
| Split | stratified 80/20, random_state 42 | **identical to Model 1**, so the scores are directly comparable |

---

## 5. The duplicate-note rule (inference)
- **Fact:** the same note often appears with different specialties (one filing per specialty), and the same (note, label) pair practically never repeats (1 of 731 duplicate groups in training).
- **Rule:** if a note to classify is already in the training data with labels {A, B}, rule those out and take the best remaining score.
- **Example:** the heart-bypass (CABG) note is in training as *Cardio*. Its test copy gets *Cardio* ruled out → *Surgery*, the model's second choice, which is correct.
- **Honesty:** it uses only training data, but it exploits how the dataset was built, not medical understanding. **Report scores with and without it** (the notebook prints both).

---

## 6. What to fill in after running (for the paper)
From the notebook output:
- [ ] Section 5: median tokens per note and % truncated
- [ ] Section 8.1: plain accuracy, macro-F1, worst class
- [ ] Section 8.2: the same numbers with the duplicate rule, plus how many test notes appear in training
- [ ] Section 8.3: full-system scores (+ keyword rule) and **BERT on notes without keywords**: the key table for the paper
- [ ] Section 8.2: per-class report (precision, recall, F1)
- [ ] Section 8.3: confusion matrix figure
- [ ] Section 9: number of mistakes and how many are "other valid label"; pick 3+ genuine errors for the Discussion
- [ ] Compare with Model 1 on the same split (Model 1: 0.442 accuracy without the rule)

---

## 7. Likely oral-exam questions

| Question | Short answer |
|---|---|
| What is fine-tuning? | Continuing to train a pre-trained model on a specific task with a small learning rate, so it adapts its knowledge instead of learning from scratch. |
| Why Bio_ClinicalBERT and not plain BERT? | Domain match: it was pre-trained on clinical notes, so it knows medical vocabulary and abbreviations. |
| What is [CLS]? | A special first token whose final vector summarises the input. The classifier reads only that vector. |
| What does self-attention do? | Each token builds its new vector as a weighted mix of all tokens' vectors (Query/Key/Value), so meaning depends on context. |
| Why no stemming here but stemming in Model 1? | BERT has its own subword tokenizer and was pre-trained on raw text. Altering the text hurts it. TF-IDF benefits from merging word forms. |
| What happens to notes longer than 512 tokens? | They are truncated. We order the fields so only the end of the transcription is cut. |
| Why weighted loss? | Classes are imbalanced (893 vs 25 notes). The weights make errors on rare classes cost more. |
| Why warm-up? | The classification layer starts random. Large early updates would damage the pre-trained weights. |
| Why lr 2e-5 and not 1e-3? | A large rate would overwrite what BERT learned in pre-training ("catastrophic forgetting"). |
| Why 4 epochs? | The recommended range for BERT fine-tuning. On a small dataset, more epochs memorise the training notes. |
| Is the duplicate-note rule cheating? | It uses only training data and is fully disclosed. We report results with and without it. |
| Isn't the keyword rule cheating? | The brief allows using all given fields, and the keyword is in the test input. It's a leak in the corpus, which we document and quantify. BERT is evaluated separately on notes without keywords. |
| How many test answers come from BERT? | 78 of 398. The other 320 come from the keyword rule. |
| What is bf16? | 16-bit floating point with the same range as 32-bit. Faster and lighter, and stable for training. |

---

## 8. Limitations and Future Work
- **512-token truncation:** chunking or long-context models.
- **Contradictory training labels:** identical notes with different labels give the model conflicting lessons. **Soft labels** (train on the note's label distribution, related to T10's "soft targets") would model the ambiguity explicitly.
- **Single run:** average several seeds and report mean ± std (T4: statistical testing).
- **Cost:** full fine-tuning updates 110M parameters. **LoRA / adapters** (T10) would train ~1% of them.
- **Other models:** PubMedBERT, BioBERT, or larger clinical encoders.

---

## 9. References for the paper
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Alsentzer et al. (2019). *Publicly Available Clinical BERT Embeddings.* Clinical NLP Workshop. (Model: `emilyalsentzer/Bio_ClinicalBERT` on Hugging Face)
- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Libraries: PyTorch, Hugging Face Transformers, scikit-learn.
- Declare the use of an LLM (Claude) to help write and explain the code.
