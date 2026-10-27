# GreenPod Labs Assignment 2 — Image Classification Using the Provided Orange Dataset

## Important dataset note
The Google Drive link supplied to the candidate downloads **Orange Dataset.zip**, not an avocado dataset. Inspection shows four image classes:
- `blackspot`
- `canker`
- `fresh`
- `grenning`

This package therefore implements a four-class **orange image condition classification** project on the dataset actually provided. It does **not** claim to classify avocado ripening stages. Confirm this dataset mismatch with GreenPod Labs before final submission, because the written assignment brief specifies avocados.

## Two required approaches
1. **Classical CV + SVM** — HSV and CIELAB color statistics plus LBP texture histogram, standardized and classified using an RBF SVM.
2. **MobileNetV2 transfer learning** — ImageNet-pretrained lightweight CNN with a new four-class classification head, augmentation and validation-loss checkpoint selection.

## Included data
The uploaded dataset has been included under `data/original_dataset/` in the ZIP, retaining its supplied `train/` and `test/` folders. The source split counts are:
- train: blackspot 184, canker 179, fresh 281, grenning 347
- test: blackspot 22, canker 22, fresh 33, grenning 22

Counts are generated from the uploaded ZIP's image files. The class imbalance should be discussed in the report. The official test folder is preserved; validation is split from the supplied training data.

## Windows setup
Open PowerShell in the extracted project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands without activating the environment using `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then replace `python` in commands with `.venv\Scripts\python.exe`.

## Run commands in order
```powershell
python src/01_eda.py
python src/02_split_dataset.py
python src/03_train_classical.py
python src/04_train_mobilenet.py --epochs 12 --batch_size 16
```

For CPU or low RAM, try `--batch_size 8`. The first MobileNetV2 run downloads ImageNet-pretrained weights and therefore needs internet access. This session could not download those weights because external network access was unavailable, so the MobileNet training has not been executed here. On your Windows PC, run it while connected to the internet. Training time depends on hardware.

After both models train, select an image from the test set and run:
```powershell
python src/05_demo.py --image "data/split/test/fresh/example.jpg"
```
Replace the example path with an actual image filename present on your computer.

## Outputs
- `outputs/eda/`: class counts, image metadata, image-size summary, class-distribution plot and contact sheet.
- `outputs/classical/`: trained SVM, metrics JSON/CSV and confusion matrix.
- `outputs/mobilenet/`: best checkpoint, training history, metrics and confusion matrix.

## Evaluation and model choice
Report accuracy, macro precision, macro recall, macro F1 and per-class metrics. Macro F1 gives equal importance to each class and is useful because the class distribution is uneven. Review the confusion matrices to identify classes the models confuse. Measure single-image latency on the target machine and state the hardware. Do not use made-up results.

## Dataset limitations
The EDA script measures counts, image dimensions and unreadable images and creates a sample contact sheet. It cannot automatically verify semantic labels or reliably detect every duplicate or lighting bias; review images manually. Be careful not to interpret the dataset labels as avocado ripeness classes.

## Mandatory questions in report
Use `report_template.md` and fill it using measured results:
1. Production performance: actual test metrics, measured latency and limits of test-to-production generalization.
2. Drawbacks: visual overlap, lighting/background changes, possible mislabeled data, and unseen conditions.
3. Improvements: more balanced data, external validation, duplicate checks, tuning, quantization and multi-view imaging.
4. Time spent: honestly describe the actual work completed; do not invent percentages.

## Deliverables
The ZIP includes source, dataset, report template, requirements and demo. For submission, add your actual results, screenshots, GitHub/Colab links and the required walkthrough recording. Because the provided dataset differs from the written avocado assignment, confirm the expected target with the assignment owner.
