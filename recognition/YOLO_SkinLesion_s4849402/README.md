# Skin Lesion Detection with YOLO — preparation stage

Student: s4849402. COMP3710, Section 2.3, Normal difficulty.

## Objective

Localize and classify annotated ISIC 2017/2018 lesions. Compare a standard YOLO baseline against a controlled modification, investigating lesion recall versus false alarms and computational cost. No real-data training results are available yet.

## Current workflow

Implementation update (2026-10-29): model adaptation, training, prediction, evaluation and official-archive import are implemented. See `RUN_GUIDE.md`. Nine logic tests and a synthetic CPU integration run passed, including a training epoch, checkpoint reload and validation APIs. These are software checks, not results on ISIC.

The student reports confirmation that the official ISIC splits may be used (2026-10-29). Official-split mode records this evidence without inventing patient IDs or claiming independent patient-identity verification.

Edit and commit code locally in VS Code on `topic-recognition`; push to the personal fork. The GPU Lab machine clones/pulls the same branch and runs training. Download raw datasets directly to a separate GPU storage directory. Download results and back up weights separately before ending a GPU session. Storage persistence has not yet been confirmed.

Local data tools require NumPy and Pillow. They are already present in the inspected local Python; local PyTorch is not required for the preparation stage.

From this directory:

```bash
python check_environment.py
```

On the GPU machine, after activating the training environment:

```bash
python check_environment.py --training
```

## Data plan

Initial diagnosis audit completed: training 2,000 images and validation 150 images, three classes, no overlapping image IDs. Neither diagnosis CSV includes patient identifiers, so patient independence remains unverified. See `reports/label_audit.md` and `reports/label_audit.json`. Reproduce with `python audit_labels.py`; this downloads only two small metadata files and does not inspect test labels.

Real-data development check: `python prepare_samples.py` retrieves the official training mask archive and one training image per class through the Archive API. Three image/mask pairs passed dimension and visual overlay inspection. See `reports/sample_audit.md` for the figure, sources and limitations. These are ground-truth boxes, not model predictions. No patient identifier was found in the three sampled API records; full patient mapping is still unresolved.

Use ISIC 2017 image data, Task 1 lesion masks and Task 3 diagnosis CSVs. Task 2 ground truth is not used. Match image IDs across sources. Use the confirmed official-split protocol and document the lack of patient metadata; never invent patient IDs.

Start by auditing training/validation metadata and sample images. Reserve test data for final evaluation after hyperparameters and confidence thresholds have been selected on validation data. Exact duplicate checks do not replace near-duplicate and patient checks.

### Manifest export

`dataset.py` accepts a CSV with `image_id,image_path,mask_path,patient_id,class_id,split` columns. Paths resolve relative to the CSV. Default `patient-id` mode requires supplied patient IDs. Alternative `official-isic` mode requires confirmation evidence and official diagnosis CSVs to check split/class membership; no patient IDs are fabricated. Train and val must be nonempty; test is optional. Use `prepare_isic.py` for extracted official archives.

```bash
python dataset.py --manifest /path/to/manifest.csv --output /path/to/new_export --names melanoma seborrheic_keratosis nevus
```

The class order above is an example: map diagnoses explicitly before export. The script validates mask alignment, rejects missing patient IDs, split leakage and decoded duplicate images, and converts the union of nonzero mask pixels to one normalized detection box. This assumes one annotated lesion per image. Review disconnected mask components visually. Empty masks require separate review, not automatic benign labels.

## Planned experiments and report

The 640 and 960 GPU runs have now completed according to the student's terminal outputs. See `reports/PROJECT_PLAN.md` for the reported validation comparison and remaining tasks, and `reports/VALIDATION_CASE_REVIEW.md` for five visually reviewed 960 validation examples. These are validation findings; final test evaluation and import of the original school-side experiment files remain pending.

See `RUN_GUIDE.md` for the implemented entry points and `CODE_BLUEPRINT.md` for the initial interface design. The planned first comparison fixes YOLOv8n and changes input size from 640 to 960 only; real-data training remains pending resource checks.

- Baseline: pretrained YOLOv8n with standard fine-tuning. Proposed experiment: change resolution or augmentation one factor at a time under identical splits and budgets. Confirm comparison and pretrained adaptation expectations with tutor.
- Report mAP50, mAP50:95, per-class precision/recall and localization IoU; define IoU aggregation and account for missed detections. Assignment target: test IoU >= 0.8, not an achieved result.
- Select confidence operating points on validation data; compare sensitivity and false positives per image.
- Record parameters, peak VRAM and inference FPS with the hardware and measurement protocol.
- Analyze 3–5 real failure cases with ground-truth/predicted boxes; state dataset and deployment limitations.
- Include the final feasibility review, exact environment versions, training commands, plots and references.

## Status

Prepared: label/sample audits, data import/export, model adaptation, training/inference/evaluation code, feasibility draft and synthetic integration checks. Official-split use is confirmed according to student feedback. Pending: full-data preparation, school GPU smoke test, real baseline/comparison experiments, final report and PR. This is not a final submission.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with interpreting the assessment and drafting the initial project plan, environment checker, mask-to-box exporter and feasibility review. Synthetic checks of box geometry, empty masks, patient leakage, duplicate images and export passed during initial preparation. These checks do not establish correctness on real ISIC data. The student must review and understand the implementation, inspect real samples and verify empirical results; update this disclosure as the work develops.

Codex subsequently implemented model adaptation, official-split import, training, prediction, evaluation and tests. A local CPU integration run used synthetic images only. See `reports/software_smoke.json` for exercised interfaces; it is not evidence of clinical performance.

## References

- COMP3710_Report_2026_v2.5.pdf, Sections 2.3 and 3–6.
- ISIC 2017 dataset: https://challenge.isic-archive.com/data/#2017
- Codella et al., Skin Lesion Analysis Toward Melanoma Detection: ISBI 2017 Challenge, https://arxiv.org/abs/1710.05006
- Ultralytics YOLOv8 documentation: https://docs.ultralytics.com/models/yolov8/
