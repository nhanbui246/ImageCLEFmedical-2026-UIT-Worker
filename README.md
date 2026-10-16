# UIT_Worker: ImageCLEFmedical 2026 Concept Detection

## 1. Introduction
This repository contains the official source code of the UIT_Worker submission to the ImageCLEFmedical 2026 Concept Detection task. Our system formulates medical concept detection as a large-scale multi-label classification problem over 2,646 Unified Medical Language System (UMLS) Concept Unique Identifiers (CUIs).

## 2. Official Result
| Item | Value |
|------|-------|
| **Task** | ImageCLEFmedical 2026 - Concept Detection |
| **Team** | `UIT_Worker` |
| **Official leaderboard team name** | `UIT_Worker` |
| **Final model** | DenseNet-121 + ImageNet + Asymmetric Loss + Full-Fit |
| **Primary F1-score** | **0.5725** |
| **Secondary F1-score** | **0.9533** |
| **Official Ranking** | **3rd place** (`final leaderboard`) |

## 3. Core Technical Methodology

* **Architecture:** DenseNet-121 initialized with standard ImageNet weights.
* **Loss Function:** Asymmetric Loss (ASL) optimized for extreme long-tail label imbalance ($\gamma^- = 4.0$, $\gamma^+ = 1.0$, clipping $c = 0.05$).
* **Training Strategy:** Full-Fit training on all 116,604 development images to maximize exposure to rare concepts.
* **Optimization:** AdamW with Cosine Annealing scheduler and PyTorch Automatic Mixed Precision (AMP) to manage GPU memory constraints.

## 4. Requirements and Installation

The codebase is written in Python (tested on Python 3.12) and utilizes PyTorch. To install the necessary dependencies, run the following command:

```bash
pip install torch torchvision timm albumentations scikit-learn numpy pandas matplotlib seaborn opencv-python tqdm
```

## 5. Data Preparation
> **Note:** The ImageCLEFmedical 2026 dataset is **not** distributed with this repository. Please obtain the official development and test datasets from the ImageCLEFmedical organizers before running the notebook.

The source code is organized as a sequential Jupyter Notebook. Before executing the pipeline, configure the correct data paths for your local or cloud environment.

### 5.1. Training Data
In **Cell 2**, update the following constants to point to the ImageCLEFmedical development dataset:
* `CSV_PATH`: Path to the `concepts.csv` file.
* `IMAGE_DIR`: Path to the directory containing the training/validation images.

### 5.2. Testing Data
In **Cell 6**, update the path to the official hidden test dataset:
* `TEST_DIR`: Path to the test images directory.

### 5.3. Model Checkpoint

After completing the training process (Cell 5), the notebook automatically saves the best-performing model checkpoint (`best_model_densenet121.pth`). Before running inference in **Cell 6**, update the checkpoint loading path to point to the saved checkpoint:

* Update:
  `torch.load("path/to/best_model_densenet121.pth", map_location=device)`

## 6. Execution Guide

The `CLEF_2026_Source_Code_UIT_Worker.ipynb` notebook is designed to be executed sequentially.

1. **Data Exploration (Cell 1):** Loads the annotation CSV, parses the UMLS CUIs, calculates statistics, and visualizes the long-tail frequency distribution.
2. **Data Pipeline (Cells 2-3):** Initializes the custom `MedicalDataset`, applies lightweight transformations via Albumentations (ShiftScaleRotate, RandomBrightnessContrast), and prepares the PyTorch `DataLoader`.
3. **Model Initialization (Cell 4):** Constructs the DenseNet-121 backbone via `timm`, sets up the custom `AsymmetricLossOptimized` criterion, and configures the AdamW optimizer.
4. **Full-Fit Training (Cell 5):** Executes the training loop for 6 epochs using all development images. Automatic Mixed Precision (AMP) with `GradScaler` is used to maintain throughput. The best model checkpoint (`best_model_densenet121.pth`) is saved automatically.
5. **Inference & Submission (Cells 6-7):** Loads the best model checkpoint generated during training, runs batch predictions on the test set, applies a fixed decision threshold of `0.60`, decodes the binary predictions back to CUI strings, and generates the final `submission.csv` mapped to the official IDs.

## 7. Citation
If you find this repository or our full-fit DenseNet-121 strategy helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{uit_worker_imageclef2026,
  author    = {Thai Q. To and Dinh C. Pham and Nhan H. Bui and Thien B. Nguyen-Tat},
  title     = {{UIT_Worker} at {ImageCLEFmedical} 2026: {DenseNet-121} Full-Fit Medical Concept Detection with Asymmetric Loss},
  booktitle = {Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2026)},
  series    = {CEUR Workshop Proceedings},
  volume    = {4283},
  pages     = {4042--4054},
  publisher = {CEUR-WS.org},
  year      = {2026},
  issn      = {1613-0073},
  url       = {https://ceur-ws.org/Vol-4283/paper297.pdf}
}
