

Project doc for AI coder: General Music PDF to MusicXML Software using AutoResearch

1. Project goal

Build a general software system that converts music PDFs into MusicXML.

The software must work on:
	•	scanned PDFs
	•	image-based PDFs
	•	digitally exported PDFs
	•	single-staff and multi-staff notation
	•	lead sheets and more fully notated pages
	•	eventually many publishers and engraving styles

Our initial supervised dataset is:
	•	a scanned Charlie Parker Omnibook PDF
	•	about 50 aligned MusicXML files from the same book

These 50 XML files are not the final scope. They are the first seed dataset for:
	•	supervised training
	•	evaluation
	•	bootstrapping alignment
	•	building synthetic tasks
	•	testing the research loop

2. Why use autoresearch here

Use karpathy/autoresearch as a research engine template, not as-is.

What we want from it:
	•	a small repo with a fixed eval harness
	•	a clear editable research surface
	•	repeated experiments by an AI coder
	•	automatic logging of results
	•	branch-based experiment history
	•	keep/discard behavior based on measurable improvement

What we do not want:
	•	to force this project into the original “single train.py only” shape if that harms the problem
	•	to optimize only one scalar metric too early
	•	to treat this as just LLM training

The repo’s core idea is still exactly useful here: fixed evaluation, constrained change surface, repeated bounded experiments, and human-written program.md instructions for the coding agent.  ￼

3. Problem framing

This is really an Optical Music Recognition (OMR) + structured decoding problem.

A strong general system should likely be split into stages:
	1.	PDF/page ingestion
	2.	page image normalization
	3.	staff/system detection
	4.	symbol detection / token recognition
	5.	musical structure reconstruction
	6.	MusicXML generation
	7.	validation / repair
	8.	confidence scoring + human review tools

Do not start with “end-to-end model from raw PDF to final XML” as the first version unless experiments clearly beat modular baselines.

4. Product definition

Input
	•	PDF file
	•	optional page range
	•	optional mode:
	•	lead sheet
	•	monophonic melody
	•	piano / grand staff
	•	auto-detect

Output
	•	MusicXML file
	•	optional per-page JSON intermediate representation
	•	optional debug overlays
	•	confidence report
	•	error flags for uncertain measures/symbols

Success criteria

A good result means:
	•	notes are correct
	•	rhythms are correct
	•	measures align
	•	pitch spelling is reasonable
	•	rests/ties/slurs are structurally valid
	•	repeats, endings, tuplets, pickup bars, and chord symbols are preserved when possible
	•	output MusicXML loads successfully in notation software

5. Initial scope and non-scope

Phase 1 scope

Focus on:
	•	monophonic melody extraction
	•	lead sheet style pages
	•	chord symbols if feasible
	•	1-staff systems first
	•	scanned and clean digital PDFs

Phase 1 non-scope

Do not prioritize initially:
	•	handwritten scores
	•	orchestral full scores
	•	very dense polyphony
	•	perfect articulation/dynamics coverage
	•	highly unusual twentieth-century notation
	•	tablature

6. Data assets we have now

We have:
	•	one scanned Charlie Parker Omnibook PDF
	•	about 50 MusicXML files from that same book

These should be used to build:

A. aligned page-to-XML dataset

Need a mapping:
	•	XML file ↔ tune title
	•	tune title ↔ PDF page(s)
	•	page region ↔ staff/system region if possible

B. supervised training set

From the 50 known pairs:
	•	page image
	•	optional cropped system image
	•	target symbolic sequence
	•	target MusicXML
	•	optional normalized intermediate token sequence

C. evaluation set

Split into:
	•	train
	•	dev
	•	test

Keep the test set frozen.

7. Critical data engineering task

Before model work, solve alignment.

The Omnibook PDF and 50 XML files must be aligned into clean pairs.

Build a preprocessing pipeline that:
	1.	rasterizes each PDF page
	2.	detects systems/staves
	3.	associates each XML file with the right page/system
	4.	creates a manifest like:

{
  "id": "ornithology_p1_sys2",
  "pdf_path": "data/raw/omnibook.pdf",
  "page_index": 14,
  "crop_bbox": [x1, y1, x2, y2],
  "xml_path": "data/xml/ornithology.xml",
  "title": "Ornithology",
  "split": "train"
}

This alignment layer is foundational. If it is weak, everything downstream will look worse than it really is.

8. Recommended internal representation

Do not train directly against raw MusicXML at first.

Use a simpler canonical music token representation first, then convert to MusicXML.

Recommended token schema, example:

CLEF_G
KEY_C
TIME_4_4
MEASURE_START
NOTE_C5_EIGHTH
NOTE_D5_EIGHTH
NOTE_Eb5_QUARTER
REST_EIGHTH
BARLINE
...

Or better, a structured event format:

{
  "measure": 12,
  "events": [
    {"type": "note", "pitch": "C5", "dur": "eighth"},
    {"type": "note", "pitch": "D5", "dur": "eighth"},
    {"type": "note", "pitch": "Eb5", "dur": "quarter"}
  ]
}

Then build:
	•	tokenizer / event vocabulary
	•	deterministic conversion from event sequence → MusicXML

This usually makes learning, debugging, and evaluation much easier than raw XML generation.

9. Architecture recommendation

Start with a modular research stack.

Baseline pipeline

PDF/Image → vision encoder → token decoder → canonical event sequence → XML renderer

Possible variants:
	•	CNN/ViT encoder + seq decoder
	•	DETR-like symbol detector + rule-based assembler
	•	image-to-sequence transformer
	•	hybrid system with staff segmentation first

My recommendation for v1

Build three baselines in order:

Baseline A: rule-heavy classical OMR starter
	•	page binarization
	•	staff line detection
	•	connected components / symbol candidates
	•	simple melody-only reconstruction

Purpose:
	•	quick reality check
	•	gives debugging visuals
	•	creates pseudo-labels or region proposals

Baseline B: cropped system image → event sequence model
	•	input: one system crop
	•	output: canonical event sequence
	•	strongest first ML baseline

Baseline C: page-level model with system segmentation
	•	only after B is stable

10. How to adapt autoresearch to this project

The original autoresearch keeps only one editable file and one fixed eval harness. We should preserve the spirit, but adapt the structure. In Karpathy’s repo, the agent is told to read the repo, keep evaluation fixed, log results, and iterate in a fresh branch.  ￼

Proposed repo layout

music-pdf2xml/
  README.md
  program.md
  data_manifest/
    dataset.csv
    splits.json
  src/
    prepare_data.py
    render_targets.py
    eval.py
    xml_writer.py
    experiments/
      train.py
      model.py
      decode.py
  outputs/
  results.tsv

Fixed files

These should be treated like prepare.py in the original repo:
	•	prepare_data.py
	•	eval.py
	•	xml_writer.py
	•	frozen test split
	•	evaluation script
	•	dataset manifest format

Editable research surface

Primary editable area for the agent:
	•	src/experiments/train.py
	•	src/experiments/model.py
	•	maybe src/experiments/decode.py

Do not let the agent constantly change the evaluation script, test split, or ground-truth conversion logic.

program.md role

program.md should instruct the agent:
	•	what files are fixed
	•	what files are editable
	•	what metric to optimize
	•	how to log experiments
	•	how to decide keep/discard
	•	complexity budget
	•	when to prefer simpler models

11. Metrics

We need more than one metric.

Primary metrics
	1.	symbol/event accuracy
	2.	note pitch accuracy
	3.	rhythm accuracy
	4.	measure validity
	5.	MusicXML parse success
	6.	sequence edit distance
	7.	full-score exact match rate for small examples

Secondary metrics
	•	chord symbol accuracy
	•	key signature accuracy
	•	time signature accuracy
	•	tie/slur accuracy
	•	accidental accuracy
	•	barline count accuracy

Practical combined score

For research selection, define one scalar:

score = 
0.30 * event_f1 +
0.25 * pitch_acc +
0.20 * rhythm_acc +
0.15 * measure_validity +
0.10 * xml_parse_rate

Then log the component metrics too.

Do not let the project optimize only XML parse rate. A file can parse and still be musically wrong.

12. Evaluation protocol

Keep evaluation fixed and frozen.

Eval set design

Split by tune, not random crop only:
	•	train: 70%
	•	dev: 15%
	•	test: 15%

Avoid leakage across systems from the same tune if possible.

Levels of evaluation
	•	token/event level
	•	measure level
	•	whole-piece level
	•	XML validity level

Error categories to log

For each run:
	•	pitch substitution
	•	octave error
	•	duration error
	•	missing rest
	•	extra note
	•	barline shift
	•	accidental omission
	•	tuplet failure
	•	tie/slur failure

13. Data augmentation strategy

Since we only have 50 XML pairs now, augmentation is essential.

Synthetic augmentation from XML

Render the 50 XML files into many notation styles:
	•	different staff thickness
	•	different spacing
	•	different DPI
	•	noise
	•	skew
	•	blur
	•	scanner artifacts
	•	contrast shifts
	•	page curvature simulation
	•	cropping / margin variation

This is probably the highest-value move.

Why

It lets the model learn:
	•	symbol invariance
	•	engraving style variation
	•	scan degradation robustness

Important rule

Always distinguish:
	•	real scanned data
	•	synthetic rendered data

Do not mix them blindly without tracking provenance.

14. Training strategy

Stage 1

Train on synthetic renders generated from the XML files.

Stage 2

Fine-tune on real scanned page/system pairs from Omnibook.

Stage 3

If needed, use pseudo-labeling on additional unlabeled music PDFs.

Stage 4

Expand to broader corpora later.

15. Generalization strategy

Because the final goal is general software, not Parker-only software, every design choice should be checked against this question:

Does this make the model better at music notation in general, or only at one book’s engraving quirks?

To reduce overfitting to Omnibook:
	•	use synthetic re-rendering in multiple fonts/styles
	•	normalize title/composer text away
	•	crop to staff systems, not whole decorative page context
	•	include augmentation that changes page look but preserves notation
	•	avoid hand-tuned Parker-specific assumptions in decoding

16. Recommended milestone plan

Milestone 0: dataset and tooling

Deliverables:
	•	PDF rasterizer
	•	page/system cropper
	•	XML-to-canonical-token converter
	•	canonical-token-to-MusicXML writer
	•	aligned manifest for 50 examples
	•	frozen train/dev/test split

Milestone 1: baseline evaluator

Deliverables:
	•	eval script
	•	XML validity checker
	•	pitch/rhythm/event metrics
	•	visual diff tool
	•	results.tsv logging

Milestone 2: first ML baseline

Deliverables:
	•	system image → event sequence model
	•	train/infer scripts
	•	dev set benchmark
	•	error analysis notebook

Milestone 3: autoresearch integration

Deliverables:
	•	program.md
	•	bounded experiment loop
	•	keep/discard rule
	•	branch naming
	•	run log format
	•	experiment dashboard

Milestone 4: product wrapper

Deliverables:
	•	CLI:
	•	musicpdf2xml input.pdf --out outdir
	•	optional GUI/web UI later
	•	confidence report
	•	debug overlay exports

17. Suggested experiment loop for autoresearch

Use autoresearch mainly on the ML baseline.

Each experiment should vary only a few things:
	•	encoder type
	•	decoder type
	•	loss weighting
	•	augmentation policy
	•	tokenization
	•	crop policy
	•	beam search / decoding constraints
	•	curriculum schedule

Example experiment policy

Per run:
	•	fixed seed set
	•	fixed train budget
	•	fixed dev set
	•	fixed eval script
	•	one branch per run tag
	•	log results to results.tsv

Like the original repo, the agent should log the result, keep/discard based on dev improvement, and prefer simpler changes when gains are tiny.  ￼

18. Example results.tsv

commit	score	event_f1	pitch_acc	rhythm_acc	xml_parse_rate	status	description
a1b2c3d	0.8123	0.8451	0.8612	0.7904	0.9800	keep	ViT-small encoder with stronger skew augmentation
d4e5f6g	0.8010	0.8320	0.8501	0.7815	0.9770	discard	added deeper decoder, no real gain
h7i8j9k	0.0000	0.0000	0.0000	0.0000	0.0000	crash	beam search decode bug

19. Very important implementation rule

The software should produce intermediate artifacts for debugging:
	•	page image
	•	detected systems
	•	token prediction
	•	reconstructed measure objects
	•	final XML
	•	diff against ground truth

Without this, the AI coder will struggle to know whether the failure is:
	•	vision
	•	symbol decoding
	•	duration reconstruction
	•	MusicXML writing
	•	evaluation bug

20. First practical target

The first truly useful target is:

Given a cropped monophonic system from a scanned PDF, output a valid MusicXML melody line with correct pitch and rhythm.

That is a real milestone.
It is narrow enough to build.
It still supports the future general software.

21. What the AI coder should build first

Tell the coder to do this in order:
	1.	ingest PDF and rasterize pages
	2.	detect/crop systems manually or semi-automatically
	3.	parse the 50 XML files into canonical event sequences
	4.	align XML files with page/system crops
	5.	create dataset manifest
	6.	build deterministic event-sequence → MusicXML writer
	7.	build evaluation harness
	8.	train first system-crop → event-sequence model
	9.	only then integrate autoresearch loop

22. What not to do

Do not:
	•	start by training on raw full pages with no alignment
	•	generate raw MusicXML strings as the only target
	•	let the agent modify evaluation and training logic at the same time
	•	mix test data into synthetic augmentation
	•	hardcode Parker-specific rules into the core decoder
	•	judge progress by visual inspection alone

23. Draft instruction for program.md

You can give the coder this direction for the project’s program.md:

# music-pdf2xml autoresearch

This project uses an autoresearch-style loop for improving a music OCR model.

## Goal
Improve dev-set score for system-image -> canonical-event-sequence transcription, then render to valid MusicXML.

## Fixed files
- src/prepare_data.py
- src/eval.py
- src/xml_writer.py
- data_manifest/splits.json
- test set and evaluation metrics

Do not modify these unless the user explicitly asks.

## Editable files
- src/experiments/train.py
- src/experiments/model.py
- src/experiments/decode.py

## Rules
- Optimize dev score, not test score.
- Keep changes small and interpretable.
- Prefer simpler code when gains are marginal.
- Log every run to results.tsv.
- Mark each run as keep, discard, or crash.
- Do not introduce Parker-specific hacks unless clearly isolated and approved.

## Metrics
Primary score is a weighted combination of:
- event F1
- pitch accuracy
- rhythm accuracy
- measure validity
- XML parse rate

## Workflow
1. Run baseline first.
2. Change one focused thing.
3. Train/evaluate.
4. Log result.
5. Keep only if score improves meaningfully or code becomes clearly simpler.

24. Final recommendation

Use the Omnibook assets as a seed lab, not as the final world.

The real product architecture should be:
	•	general input pipeline
	•	canonical symbolic representation
	•	deterministic XML writer
	•	strong eval harness
	•	autoresearch loop only around the model and decoding layer

That gives you something much more durable than “Parker PDF converter.”
