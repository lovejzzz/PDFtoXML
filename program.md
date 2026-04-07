# music-pdf2xml autoresearch

This project uses an autoresearch-style loop for improving a music OCR model.

## Goal
Improve dev-set score for system-image → canonical-event-sequence transcription, then render to valid MusicXML.

## Fixed files (do not modify unless user explicitly asks)
- `src/types.py` — canonical event schema
- `src/prepare_data.py` — MusicXML parser
- `src/eval.py` — evaluation harness
- `src/xml_writer.py` — events → MusicXML writer
- `src/ingest_pdf.py` — PDF rasterizer
- `data_manifest/splits.json` — frozen train/dev/test split
- `data_manifest/manual_page_map.json` — page alignment

## Editable files (research surface)
- `src/experiments/train.py` — training loop
- `src/experiments/model.py` — model architecture
- `src/experiments/decode.py` — decoding and post-processing
- `src/experiments/dataset.py` — data loading and augmentation
- `src/experiments/runner.py` — experiment orchestration

## Experiment rules

### Keep/discard
- **Keep** if dev score improves by ≥ 0.005 over current best
- **Discard** if improvement < 0.005
- **Prefer simpler code** when gains are marginal (< 0.01)
- Mark each run as `keep`, `discard`, or `crash` in results.tsv

### Complexity budget
- Change ONE focused thing per experiment
- Keep changes small and interpretable
- Do not introduce Parker-specific hacks
- Do not optimize test score — only dev score

### Logging
- Every run logged to `results.tsv` (tab-separated)
- Detailed experiment log in `outputs/experiments_log.json`
- Format: `commit score event_f1 pitch_acc rhythm_acc measure_validity xml_parse_rate status description`

### Experiment naming
- Descriptive snake_case: `wider_d512`, `lr_3e4`, `dropout_02`
- Name reflects the ONE thing that changed

## Metrics

Primary score (weighted combination):
- `event_f1` × 0.30 — F1 using exact event matching
- `pitch_acc_global` × 0.25 — correct pitch / total gold notes
- `rhythm_acc_global` × 0.20 — correct duration / total gold notes
- `measure_validity` × 0.15 — fraction of metrically valid measures
- `xml_parse_rate` × 0.10 — parseable output fraction

## Workflow
1. Check current best score in results.tsv
2. Pick ONE focused change from the search space
3. Train and evaluate on dev set
4. Log result to results.tsv
5. Keep only if score improves meaningfully (≥ 0.005)
6. If kept, save checkpoint and commit
7. If discarded, revert and try next

## Running experiments
```bash
# Run all pending experiments
python -m src.experiments.runner

# Run a specific experiment
python -m src.experiments.runner --run wider_d512

# View results dashboard
python -m src.experiments.runner --dashboard

# List all configs
python -m src.experiments.runner --list
```

## Current baseline
- CNN encoder (4-block) + Transformer decoder (4 layers, d=256)
- 3.7M parameters
- Trained on 560 synthetic images + 35 real (two-stage)
- Best dev score: 0.1797

## Data
- 50 aligned MusicXML files from the Charlie Parker Omnibook
- Canonical event representation with 48 divisions per quarter note
- Frozen train/dev/test split by tune (35/7/8, seed=42)
- 750 synthetic images (15 augmentation configs × 50 tunes)
