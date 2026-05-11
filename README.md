# Hydrological Event Detection from Social Media

Code for the paper:

> **A Machine Learning Approach to Hydrological Event Detection from News-informed Social Media Alerts**
> Pita Costa J., Corzo Perez G., Topal O., Mikoš M., Novalija I., Orel R., Casals del Busto I., Goveas N.
---

## What this does

Extracts a 7-dimensional feature vector from each tweet using zero-shot prompting via the Google Gemini API, then evaluates the vectors against expert-annotated ground truth across four flood and landslide case studies.

| Feature | Range | Description |
|---|---|---|
| `hydro_relevance` | 0–1 | Is the tweet about actual water/land displacement? |
| `severity_index` | 0–1 | How life-threatening is the described event? |
| `immediacy` | 0–1 | Is the event happening right now? |
| `first_hand_witness` | 0–1 | Is the user describing what they see? |
| `locational_precision` | 0–1 | Are specific landmarks or streets named? |
| `distress_signal` | 0–1 | Are there linguistic markers of panic or emergency? |
| `event_type` | flood \| landslide \| null | Type of event, if any |

---

## Setup

```bash
pip install -r requirements.txt
```

Requires a Google Cloud project with Vertex AI enabled. Authenticate with:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Then set your project ID in `hydro_feature_extractor.py`:

```python
PROJECT_ID = "your-project-id"
LOCATION   = "us-central1"
```

---

## Usage

**Extract features:**
```bash
python hydro_feature_extractor.py your_tweets.csv
# → your_tweets_features.jsonl
```
Interrupted runs resume automatically. Recover failed rows with:
```bash
python retry_failed.py your_tweets_features.jsonl
```

**Sample for annotation:**
```bash
python annotation_sampler.py your_tweets_features.jsonl --stats  # check distribution first
python annotation_sampler.py your_tweets_features.jsonl          # generate CSVs
# → _annotate_blind.csv  (fill in expert_label: 1 = disaster, 0 = not)
# → _annotate_full.csv   (feature scores for comparison)
```

**Evaluate against labels:**
```bash
python evaluate_labels.py annotate_full.csv annotate_blind.csv
```

---

## Input format

CSV with at minimum:

| Column | Description |
|---|---|
| `body` | Tweet text |
| `dateTimePub` | Timestamp (`Jul 31, 2021 @ 16:31:57.000` or ISO format) |
| `tweet_id` | Unique identifier (recommended) |

---

## Case studies

| Dataset | Event | Tweets | HIGH (≥0.30) | MIDDLE | LOW (<0.05) |
|---|---|---|---|---|---|
| Germany July 2021 | European floods | 3,198 | 35.8% | 22.3% | 41.9% |
| India August 2020 | Pettimudi landslide | 3,753 | 15.9% | 9.8% | 74.3% |
| Japan July 2021 | Atami landslide | 1,101 | 2.6% | 4.0% | 93.4% |
| Indonesia July 2020 | North Luwu floods | 9,663 | 0.9% | 1.4% | 97.7% |

Tweet data was collected under the NAIADES research agreement and cannot be redistributed. Annotated evaluation subsets (50 tweets per case study) are included in `annotations/`.

---

## Citation

```bibtex
@article{pitacosta2026hydro,
  title   = {A Machine Learning Approach to Hydrological Event Detection from News-informed Social Media Alerts},
  author  = {Pita Costa, Joao and Corzo Perez, Gerald and Topal, Oleksandra and Mikoš, Matjaž and Novalija, Inna and Orel, Rok and Casals del Busto, Ignacio and Goveas, Neena},
  journal = {Water},
  year    = {2026},
}
```

Code released under MIT Licence. Twitter data subject to the Twitter/X Developer Agreement.
