# PDFtoXML Project Log

## Goal
Build a general software system that converts music PDFs into MusicXML, starting with the Charlie Parker Omnibook as seed data.

## What We Built (April 7-14, 2026)

### Infrastructure (Milestones 0-1)
- **PDF rasterizer**: 145 pages at 300 DPI from scanned Omnibook
- **MusicXML parser**: 50 ground-truth XMLs → canonical event sequences (22,699 notes, 48-division grid)
- **Evaluation harness**: event F1, pitch/rhythm accuracy, measure validity, combined score
- **XML writer**: canonical events → MusicXML 3.0 (round-trip: 50/50 pass, self-eval 0.9995)
- **Page-to-tune alignment**: manual page map for all 50 tunes (120 pages)
- **Train/dev/test split**: 35/7/8 tunes, frozen, seed=42

### Autoresearch Loop (Milestone 3)
- **runner.py**: bounded experiment orchestration with keep/discard rules
- **12+ experiment configs** tested in Round 1 (model size, resolution, LR, regularization)
- **results.tsv**: 30+ logged experiment runs

### Key Breakthroughs (chronological)

| Date | Score | What Changed |
|------|-------|-------------|
| Apr 7 | 0.1634 | First CNN-Transformer baseline (35 real images) |
| Apr 7 | 0.1797 | + Synthetic data (750 MuseScore renders) |
| Apr 7 | 0.1943 | + Lower resolution (256×192) |
| Apr 7 | 0.2210 | + **Scan-like augmentation** (domain adaptation breakthrough) |
| Apr 8 | 0.2344 | + **PrIMuS pre-training** (19K samples) + staff segmentation |
| Apr 8 | 0.4053 | + **Full 87K PrIMuS** pre-training (20 epochs, d=256, 4L) |
| Apr 9 | 0.4153 | + Tiny-LR continuation chain (V4→V8→V9) |
| Apr 10 | 0.4280 | + **6-layer model** (40 epochs, 7.2M params, val=0.19) |
| Apr 13 | 0.4292 | + **8-layer d=384 model** (60 epochs, 20.2M params) |

### Architecture
```
PDF Page → homr segmentation → Staff Crops (128×1024 px)
                                    ↓
                    PrIMuS Pre-trained CNN-Transformer
                    (82,650 training staves, 60+ epochs)
                                    ↓
                    Per-staff token prediction (greedy decode)
                                    ↓
                    Assembly: concatenate staves, dedup headers
                                    ↓
                    Canonical events → MusicXML output
```

### What We Tried That Didn't Work
- **Full-page recognition**: model can't learn from full pages, needs staff-level crops
- **Multi-page data expansion**: pages don't align to token subsequences
- **Pseudo-labeling**: model confidence too low (3-5%) on real scans
- **Beam search**: greedy decoding is better (beam adds extra notes)
- **Post-processing** (stutter removal, octave fix, duration enforcement): too aggressive, removes correct notes
- **Label smoothing, high dropout, weight decay**: all hurt
- **Simple ensemble** (6L+8L by selection): worse than individual models
- **Label cleaning** (filter low-similarity staves): less data always hurts

### What We Learned
1. **Data quantity > model architecture** (until 87K PrIMuS, then model size matters)
2. **Domain adaptation via augmentation** is the single biggest win for scanned images
3. **Staff-level segmentation is essential** — aligns training (PrIMuS) with inference (Omnibook)
4. **Per-staff label alignment is noisy** — measures are split heuristically, not perfectly
5. **Pre-train val loss correlates with final score**: 0.46→0.40, 0.19→0.43, lower→better
6. **Our model (0.43) beats homr (0.25)** on this dataset by 67%

### Current Best Model
- **Score: 0.4292**
- Architecture: 8 decoder layers, d=384, 6 attention heads, FFN=1536
- Parameters: 20.2M
- Pre-training: 82,650 CameraPrIMuS staves, 60 epochs
- Fine-tuning: 667 Omnibook staff crops, 60 epochs
- Pitch accuracy: 47.1%
- Measure validity: 66.8%

### Files & Structure
```
src/
  types.py              — canonical event schema (48-division grid)
  prepare_data.py       — MusicXML → canonical events
  xml_writer.py         — canonical events → MusicXML
  eval.py               — evaluation harness
  ingest_pdf.py         — PDF rasterizer
  align_data.py         — XML ↔ PDF page alignment
  extract_staffs.py     — homr-based staff crop extraction
  render_targets.py     — MuseScore synthetic rendering
  benchmark_homr.py     — homr evaluation
  experiments/
    model.py            — CNN/ResNet encoder + Transformer decoder
    dataset.py          — OMR dataset (real + synthetic + pseudo)
    staff_dataset.py    — staff crop dataset
    primus_loader.py    — CameraPrIMuS data loader
    train.py            — training loop
    train_primus.py     — 2-stage PrIMuS pretrain + fine-tune
    finetune_only.py    — fine-tune from existing checkpoint
    runner.py           — autoresearch experiment orchestrator
    scan_augment.py     — scan-like domain augmentation
    postprocess.py      — musical post-processing rules
    decode.py           — token → ScoreData conversion
    pseudo_label.py     — semi-supervised labeling
    clean_labels.py     — model-based label quality scoring
    eval_beam.py        — beam search evaluation
    eval_tta.py         — test-time augmentation
    eval_pretrained_only.py
    eval_homr_corrected.py
    eval_homr_clef_fix.py
```

## In Progress
- **80-epoch 8L d=384 pre-training** (epoch 19/80, val=0.45, converging 2x faster than prev run)
- Target: push val loss below 0.15, expecting score 0.45-0.50+

## Plan for Next Session

### Immediate (when 80-epoch run finishes)
1. Fine-tune the 80-epoch checkpoint on Omnibook staff crops
2. Run tiny-LR continuation chain (5e-6, 3e-6)
3. Eval on PrIMuS val to measure raw OMR accuracy improvement

### Short-term (next few days)
4. **Download more training data**: MUSCIMA++, DeepScoresV2 — adds 100K+ more labeled staves
5. **Improve per-staff label alignment**: use model predictions to correct noisy ground-truth splits
6. **Smarter ensemble**: token-level voting across multiple checkpoints instead of tune-level selection

### Medium-term (next week)
7. **Edit-distance based evaluation**: current exact-position eval understates true quality
8. **Musical constraint decoding**: key-signature-aware beam search that rejects impossible notes
9. **Curriculum learning**: start with short PrIMuS sequences, gradually add longer ones
10. **Train on combined PrIMuS + MUSCIMA++ + DeepScores**: 200K+ staves

### Reaching 90%+
11. **Integrate Sheet Music Transformer (SMT)**: SOTA model with pretrained weights
12. **Use LLM vision API**: send staff crops to Claude/GPT-4V for recognition
13. **Manual per-staff annotation**: curate 500+ perfectly labeled Omnibook staves
14. **End-to-end system**: PDF → segmentation → recognition → post-processing → MusicXML

## External Data
- **CameraPrIMuS**: 87,678 distorted monophonic staves (downloaded, 2.3GB)
- **homr**: installed, benchmarked (used for staff segmentation)
- **MuseScore 4**: used for synthetic rendering (750 images)
- **SMT repo**: cloned but not yet integrated

## How to Resume
```bash
cd /Users/mengyingli/PDFtoXML
source .venv/bin/activate

# Check if training is running
ps aux | grep train_primus | grep -v grep

# Check latest epoch
grep "Epoch" /private/tmp/claude-501/-Users-mengyingli/*/tasks/*.output | tail -5

# Check best score
sort -k2 -t$'\t' -nr results.tsv | head -5

# Run evaluation on a checkpoint
python -m src.experiments.eval_pretrained_only primus_finetuned.pt

# Launch new training
python -m src.experiments.train_primus pretrain_samples=87000 pretrain_epochs=80 ...
```
