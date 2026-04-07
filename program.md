# music-pdf2xml autoresearch

This project uses an autoresearch-style loop for improving a music OCR model.

## Goal
Improve dev-set score for system-image → canonical-event-sequence transcription, then render to valid MusicXML.

## Fixed files
- `src/types.py`
- `src/prepare_data.py`
- `src/eval.py`
- `src/xml_writer.py`
- `data_manifest/splits.json`
- Test set and evaluation metrics

Do not modify these unless the user explicitly asks.

## Editable files
- `src/experiments/train.py`
- `src/experiments/model.py`
- `src/experiments/decode.py`

## Rules
- Optimize dev score, not test score.
- Keep changes small and interpretable.
- Prefer simpler code when gains are marginal.
- Log every run to `results.tsv`.
- Mark each run as `keep`, `discard`, or `crash`.
- Do not introduce Parker-specific hacks unless clearly isolated and approved.

## Metrics

Primary score is a weighted combination of:
- `event_f1` (0.30)
- `pitch_acc_global` (0.25)
- `rhythm_acc_global` (0.20)
- `measure_validity` (0.15)
- `xml_parse_rate` (0.10)

## Workflow
1. Run baseline first.
2. Change one focused thing.
3. Train/evaluate.
4. Log result.
5. Keep only if score improves meaningfully or code becomes clearly simpler.

## Data
- 50 aligned MusicXML files from the Charlie Parker Omnibook
- Canonical event representation with 48 divisions per quarter note
- Frozen train/dev/test split by tune (70/15/15, seed=42)

## Architecture
- PDF → page images (`src/ingest_pdf.py`)
- MusicXML → canonical events (`src/prepare_data.py`)
- Canonical events → MusicXML (`src/xml_writer.py`)
- Evaluation harness (`src/eval.py`)
- ML experiments in `src/experiments/`
