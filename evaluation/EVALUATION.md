# VideoRAG for Courtroom Evidence — Evaluation

This document reports seven evaluations of the system: base-architecture correctness
(timestamp fidelity, multi-modal retrieval, image-to-video search) and the three
novelty layers (cross-session speaker verification, contradiction detection, citation
verification), plus a scalability/robustness measurement. All numbers were produced by
running the actual production code paths against real uploaded videos on this machine
— nothing here is simulated. Scripts are in `evaluation/scripts/`, raw metrics in
`evaluation/results/*.json`, figures in `evaluation/figures/*.png` (300 DPI, paper-ready).

**Reproduce:** `source .venv/bin/activate && python evaluation/scripts/eval_0N_*.py`
(the backend server must be running against a populated `data/` directory — see
"Test corpus" below for how to rebuild it).

## Test corpus

| Label | Content | Duration | Source |
|---|---|---|---|
| `session1`/`session2`/`session3` | Synthetic witness statements, same TTS voice, `session2` contradicts `session1`, `session3` is consistent with `session1` and contradicts `session2` | 3–6s each | Generated (gTTS + ffmpeg) for this evaluation, ground truth by construction |
| `obama` | Real public-domain speech recording | 214s | [Internet Archive, SCVTV](https://archive.org/details/SCVTV.com_8_21_2010_President_Barack_Obama_s_Weekly_Address_-), public domain |
| `detection` | Synthetic clip combining a real stock photo (bundled with `ultralytics`) and a text overlay | 3s | Constructed for OCR/detection testing |
| `clancy1`/`clancy2` | Real courtroom trial livestream coverage | 1395s / 1316s | Public YouTube livestream coverage of a criminal trial, downloaded for local testing |

**A note on scale.** This is an exploratory-scale evaluation built during active
development, not a large pre-registered benchmark — sample sizes range from n=10 to
n=697 depending on what could be measured directly from real system state versus what
required hand-authored ground truth. Every section below states its n and how ground
truth was established; treat single-digit-n results (Evals 4–6) as sanity/floor
validations, not statistically powered claims. A follow-up with the U.S. Courts
Archive's multi-session trial coverage (many videos of the same speakers, professional
reference transcripts) would let several of these graduate to a properly powered
benchmark — noted per-section below.

---

## Eval 1 — Timestamp fidelity

**What it tests:** the frame-timestamp bug fix (`timestamp = frame_idx / video_fps`,
computed from the real OpenCV frame position rather than assumed from a fixed sample
rate).

**Method:** for every video actually processed by the system, the real stored
timestamps in `image_index` are compared against what the pre-fix formula
(`saved_frame_number / FRAME_SAMPLE_FPS`) would have produced for the same frames,
using each video's real duration and frame count. This isn't a synthetic what-if — it's
the literal old formula applied to real post-fix data.

**Results** (`eval01_timestamp_fidelity.json`, figures `eval01_timestamp_fidelity.png`,
`eval01_timestamp_mae_summary.png`):

| Video | Duration | Frames | Drift vs. pre-fix formula (MAE / max) |
|---|---|---|---|
| session1/2/3, detection | 2–5s | 3–6 | 0.0s / 0.0s (bug never manifests below the old 48s assumption) |
| obama | 214.2s | 73 | 72.1s / 142.2s |
| clancy1 | 1393.4s | 300 | 546.4s / **1094.4s** |
| clancy2 | 1315.3s | 300 | 507.5s / **1016.3s** |

On the two real courtroom recordings, the pre-fix formula would have placed the last
retrieved frame's timestamp over **18 minutes off** from where it actually occurs in a
~22-minute video — exactly the "wrong clip" failure mode the fix targets, and it scales
with video length as predicted, confirmed on two independent 22-minute+ real recordings.

---

## Eval 2 — Multi-modal retrieval quality

**What it tests:** whether the ASR/caption/OCR indexes can retrieve their own content
back given a natural-language-style query — a standard self-supervised retrieval check
for an existing index with no separately hand-labeled query set.

**Method:** 40 random documents sampled per modality from the real indexes (697 speech
segments, 81 captions, 228 OCR entries — pooled across all 7 videos); each document's
first 5 words become the query, run through the production `search_*` functions;
Recall@{1,3,5,10} and MRR computed two ways — **exact-frame** (did we recover the
literal source frame) and **content-match** (did we recover *any* frame with identical
stored text, since captions/OCR are frequently near-duplicated across nearby frames and
exact-frame recall unfairly penalizes a correct-but-different match).

**Results** (`eval02_retrieval.json`, figures `eval02_recall_at_k.png`,
`eval02_mrr_strict_vs_content_aware.png`):

| Modality | Unique / total docs | Recall@1 (exact) | Recall@10 (exact) | MRR (exact) | MRR (content-aware) |
|---|---|---|---|---|---|
| Speech (ASR) | 629 / 697 | 0.43 | 0.58 | 0.47 | 0.57 |
| Caption | 29 / 81 | 0.00 | 0.15 | 0.02 | 0.03 |
| OCR | 11 / 228 | 0.00 | 0.05 | 0.02 | 0.00 |

**Honest finding, not hidden:** caption and OCR retrieval are substantially weaker than
speech retrieval, and it's traceable to a real cause, not a metric artifact — captions
are heavily templated (only 29 unique captions across 81 stored, e.g. repeated "a
detailed description of this scene showing the trial proceedings..." phrasing), and OCR
text is even more repetitive (11 unique strings across 228 entries) while apparently
too short/noisy for CLIP's text encoder to reliably re-match. The content-aware variant
doesn't rescue OCR (0.00 MRR) — the embedding isn't finding *any* matching-text frame,
not just missing the exact one. This points at two independent, actionable issues worth
naming in a limitations section: (1) caption generation producing low-diversity,
templated text rather than genuinely distinctive per-frame descriptions, and (2) CLIP's
text encoder being a poor fit for literal short-string OCR matching — a dedicated text
embedder for the OCR index specifically may be worth trying.

---

## Eval 3 — Image-to-video person search (Strategy B)

**What it tests:** given a reference photo, does the system correctly find the frames
where that person actually appears, and correctly *not* match frames of a different
scene entirely?

**Method:** a face crop taken directly from the `obama` video is scored (CLIP cosine
similarity) against every stored frame across **all 7 videos** — 73 frames genuinely
containing that person (positive class) vs. 616 frames from entirely different
videos/scenes (negative class, real discriminative task, not a toy).

**Results** (`eval03_image_search.json`, figures `eval03_image_search_roc.png`,
`eval03_image_search_pr.png`, `eval03_image_search_score_dist.png`):

- **ROC-AUC = 0.959, EER = 0.021** at threshold 0.842 — the embedding itself separates
  the two classes almost perfectly (score-distribution figure shows two clean,
  non-overlapping clusters: positives at 0.82–0.90, negatives at 0.28–0.44).
- **But at the system's actually-deployed thresholds**, precision is far lower than the
  AUC suggests: **48.3% precision** at the chat's `min_score=0.35`, and only **13.1%
  precision** at the hard confidence gate of 0.30 (both still 95.9% recall). This is a
  genuine, actionable finding, not a contradiction of the AUC number — it's a class-
  imbalance effect: the deployed thresholds sit *inside* the negative-class distribution
  rather than in the empty gap between the two clusters (roughly 0.45–0.80), so a
  majority of frames above 0.35 are false positives even though the classifier itself is
  excellent.
- **Recommendation, backed by the data above:** raise `min_score` into the 0.5–0.7
  range. The clean bimodal separation means this would move precision toward 1.0 with
  no recall cost on this test set.

---

## Eval 4 — Cross-session speaker verification (Novelty 1)

**What it tests:** does cosine similarity on Resemblyzer voiceprints correctly link the
same speaker across separate video uploads, and correctly reject different speakers?

**Method:** every voiceprint the running system actually captured (5 total —
`session1/2/3`, `obama`, and one Clancy-trial speaker; see limitation below) is compared
pairwise. `session1/2/3` share a TTS voice (positive/same-speaker ground truth by
construction); every other pair is a genuinely different real or synthetic voice
(negative ground truth).

**Results** (`eval04_speaker_verification.json`, figures
`eval04_speaker_similarity_matrix.png`, `eval04_speaker_score_by_class.png`):

- n=10 pairs (3 same-speaker, 7 different-speaker). **ROC-AUC = 1.0.**
- At the system's configured matching threshold (0.75): **precision = 1.0, recall =
  1.0** (3/3 same-speaker pairs correctly linked, 7/7 different-speaker pairs correctly
  rejected).
- Same-speaker similarities cluster at 0.92–0.95; different-speaker similarities at
  0.46–0.65 — a comfortable margin either side of 0.75.

**Limitation, stated plainly:** n=10 is small, and it's small for a specific, real
reason worth reporting — Clancy trial video 1's four diarized speakers were processed
*before* the voiceprint-extraction OOM fix (Eval 7) and never got a captured voiceprint,
so they couldn't be included as additional negative pairs. Re-processing that video
(now that the fix is in place) would extend this to n=13+ pairs; a proper benchmark
needs videos with the *same* speaker recurring across sessions, which requires a
multi-session real trial archive (the two Clancy clips here are different witnesses,
useful as negative pairs but not as a true-positive cross-session test).

---

## Eval 5 — Contradiction detection (Novelty 2)

**What it tests:** 3-way classification (CONTRADICT / ENTAIL / UNRELATED) of statement
pairs.

**Method:** 18 pairs I authored specifically for this evaluation (6 per class,
generic courtroom-testimony phrasing, unambiguous by design — not derived from any real
transcript, so labels are uncontestable) + 3 pairs produced end-to-end by the actual
system from real uploaded audio (`session1/2/3`, Eval 4's ground truth), run through the
production `check_contradiction()` LLM classifier.

**Results** (`eval05_contradiction_detection.json`, figures
`eval05_contradiction_confusion_matrix.png`, `eval05_contradiction_prf1.png`):

- n=21. **Accuracy = 100%**, precision/recall/F1 = 1.0 on all three classes.
- **CONTRADICT false-positive rate = 0.0** — worth calling out specifically, since in a
  legal-evidence tool a false contradiction flag (accusing a witness of inconsistency
  they didn't commit) is a more costly failure mode than a missed one.

**Limitation, stated plainly:** this is a clean-case sanity check, not a stress test.
The 18 hand-authored pairs were deliberately unambiguous so their ground truth would be
uncontestable — 100% accuracy here demonstrates the classifier gets clear-cut cases
right, not that it handles the genuinely subtle, partially-worded contradictions real
testimony produces. A rigorous follow-up needs real (or realistic paraphrased) courtroom
statement pairs labeled by **two independent human annotators**, with inter-annotator
agreement (Cohen's κ) reported alongside the model's accuracy — the plan this project
started from names this explicitly and it still stands as the next step.

---

## Eval 6 — Citation verification faithfulness (Novelty 3)

**What it tests:** does the post-answer verification step actually catch a citation
that doesn't support its claim, and correctly pass one that does — and how much does
that layer add over just trusting every citation?

**Method:** 10 claims I authored, each paired with a real `(video_id, timestamp)` from
the `obama` video — 5 accurately summarize what's genuinely discussed near that
timestamp (should verify as supported), 5 are deliberately swapped to an unrelated topic
at that same timestamp (should verify as unsupported). Run through the production
`verify_citation()`, scored as a binary classifier, compared against the naive "trust
every citation" baseline (i.e., what the system would do with Novelty 3 removed).

**Results** (`eval06_citation_verification.json`, figures
`eval06_citation_with_vs_without.png`, `eval06_citation_confusion_matrix.png`):

- n=10. **Verifier accuracy = 100%** (5/5 supported correctly passed, 5/5 unsupported
  correctly caught) vs. **50% for the naive baseline** — exactly what "trust everything"
  scores on a balanced set, and the clean +50 point delta is the layer's directly
  measurable contribution on this test set.

**Limitation:** as with Eval 5, this is a balanced, designed sanity set (5/5), not a
measurement of the real-world hallucination rate the system produces unprompted — that
number requires sampling real generated answers and having a human judge each citation,
which is the natural next step and was in the original evaluation plan.

---

## Eval 7 — Processing time / scalability, and the voiceprint-OOM fix

**What it tests:** how processing time scales with video duration, and whether the
Eval-4-blocking voiceprint OOM fix (Resemblyzer call capped to 45s of audio instead of
an entire session) changed anything measurable about processing cost.

**Method:** real wall-clock timestamps parsed from this project's own backend logs
across every video actually uploaded during development (n=15), duration paired with
total processing time, split into **cold** (first upload after a server restart — pays
a one-time model-load cost) vs. **warm** (later uploads in the same process, models
already cached).

**Results** (`eval07_scalability.json`, figures `eval07_scalability.png`,
`eval07_oom_fix_before_after.png`):

- **One outlier excluded from the cold-start average and reported separately**: the very
  first upload of the entire project (`64676f3b-1c5`) took 1304s not because of model
  *loading* but model *downloading* (CLIP weights over the network, first run only) —
  categorically different from ordinary cold starts and would badly skew the mean if
  folded in.
- Ordinary cold starts (n=7, excluding the download outlier): mean **59.3s** overhead
  from model loading alone, on top of near-zero actual content-dependent processing for
  short clips.
- Warm-state processing is **well under real-time even for 22-minute recordings**: the
  1316–1395s Clancy videos processed in 143–151s warm/cold respectively — roughly **9×
  faster than playback**, because frame extraction is capped by the compute budget (300
  frames) rather than growing linearly with duration, and Whisper transcription on GPU
  outpaces real-time by a wide margin.
- **OOM fix, before vs. after**: reprocessing the identical 1316s video pre- and
  post-fix cost 144s vs. 151s — a ~5% difference, i.e. **the fix is not a speed
  optimization**, and shouldn't be reported as one. Its value is purely correctness: the
  pre-fix run silently produced *zero* usable voiceprint for that speaker (caught
  exception, empty fallback), while the post-fix run captured one successfully — the
  entire cross-session identity feature for that speaker either works or doesn't,
  independent of the ~7s timing difference.

---

## Summary table for the paper

| Layer | Metric | Result | n |
|---|---|---|---|
| Base — timestamps | Max drift corrected by fix (22-min video) | 1094.4s | 2 real videos |
| Base — retrieval (ASR) | MRR | 0.47 (0.57 content-aware) | 40 queries |
| Base — retrieval (caption/OCR) | MRR | 0.02 / 0.02 (limitation, see Eval 2) | 40 queries each |
| Base — image search | ROC-AUC / EER | 0.959 / 0.021 | 689 frames |
| Novelty 1 — speaker verification | Precision/Recall @ threshold | 1.0 / 1.0 | 10 pairs |
| Novelty 2 — contradiction detection | Accuracy, CONTRADICT FPR | 1.0, 0.0 | 21 pairs |
| Novelty 3 — citation verification | Accuracy vs. naive baseline | 1.0 vs. 0.5 | 10 cases |
| System — scalability | Warm processing vs. real-time (22-min video) | ~9× faster | 15 runs |

## What would make this a stronger benchmark (for future work / limitations section)

1. **Multi-session real trial archive** (e.g. U.S. Courts Archive) with the *same*
   speakers recurring across videos — needed for a properly powered Eval 4 true-positive
   rate, beyond the current n=3.
2. **Human-annotated contradiction pairs with inter-annotator agreement** — Eval 5's
   100% is a floor on clean cases, not a ceiling on real ambiguity.
3. **Sampled real-answer citation audit** — Eval 6 measures the verifier's classifier
   accuracy on designed cases, not the system's unprompted hallucination rate.
4. **A dedicated OCR text embedder** — Eval 2 suggests CLIP's text encoder is a weak
   fit for literal short-string OCR matching; worth an ablation against a plain
   sentence embedder.
5. **DER (Diarization Error Rate)** against a manually-labeled reference RTTM — not
   attempted here due to the manual annotation cost, but standard in the diarization
   literature and expected by reviewers familiar with pyannote-based work.
