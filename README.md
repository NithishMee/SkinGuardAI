# SkinGuardAI: Deep Multi-Modal Dermoscopy Diagnostic Workstation

SkinGuardAI is an end-to-end clinical deep learning system for dermoscopic skin lesion screening and decision support, built on the **HAM10000** dataset.

---

## 🔬 System Architecture

- **Dual-Branch Feature Extraction**:
  - **CNN Branch**: ImageNet-pretrained `ConvNeXt-Tiny` capturing fine micro-structures, pigment networks, and boundary irregularities.
  - **Vision Transformer Branch**: Pretrained `Swin Transformer Tiny` modeling long-range spatial context across the lesion topography.
- **Adaptive Attention Fusion**:
  - Dynamically weights and fuses local convolutional representations and global transformer embeddings.
- **Explainable AI (XAI)**:
  - **Multi-Scale Hierarchical Layer-CAM**: Eliminates ConvNeXt gradient cancellation through spatial element-wise positive weighting ($w_{k,i,j} = \text{ReLU}(\nabla A_{k,i,j}) \cdot A_{k,i,j}$), harmonically fusing Stage 5 (14x14 micro-pigment networks) and Stage 7 (7x7 global lesion core) with selectable colormaps (Turbo, Jet, Magma).
- **Clinical Safety Gating**:
  - **Fixed &ge;90.0% Confidence Gate**: Inferences with peak class probability below 90% are flagged as **"Unknown Category"** with exploratory Layer-CAM active to guide physician biopsy site planning while preventing automated overconfidence.
- **Clinical Reporting**:
  - Direct export of structured **PDF & TXT Clinical Consultation Reports** including patient demographics, lesion site, probabilities, and diagnostic protocols.

---

## 📁 Project Structure

```text
SkinGuardAI/
├── app.py                  # Streamlit Clinical Diagnostic Workstation
├── medfusionnet.py         # MedFusionNet architecture & checkpoint loader
├── ham10000_data.py        # PyTorch Dataset, transforms, and DataLoaders
├── gradcam.py              # Multi-scale Grad-CAM++ visualization utility
├── train.py                # High-performance training pipeline (Pause/Resume)
├── evaluate.py             # Evaluation & clinical performance metric generator
├── tune.py                 # Bayesian hyperparameter optimization (Optuna)
├── prepare_ham10000.py     # Lesion-isolated split generation utility
├── open_web_interface.bat  # 1-Click launcher for local clinical UI
├── run_local_demo.bat      # CLI launcher for Streamlit interface
├── run_train.sh            # HPC training script
├── train_hpc.ipynb         # Interactive Jupyter training notebook for HPC
├── requirements.txt        # Python dependency manifest
├── dataset/                # HAM10000 images & metadata (Preserved)
├── splits/                 # Lesion-isolated train/val/test partitions
├── checkpoints/
│   └── medfusionnet_best.pt# Active calibrated best model checkpoint
├── outputs/
│   ├── training_history.csv# Training log metrics
│   └── evaluation/         # Confusion matrix and ROC curves
└── static/                 # Static asset directory for clinical reports
```

---

## 🚀 Quick Start (Local Web Interface)

### Method 1: Double-Click Launcher
Double-click `open_web_interface.bat`. It will start the server and open your default browser to `http://localhost:8501/`.

### Method 2: Command Line
```bash
# Activate your environment
.\.venv\Scripts\activate

# Launch the clinical workstation
streamlit run app.py --server.port 8501
```

---

## 🏋️ Model Training & HPC Execution

### Train Locally or on Server
```bash
python train.py --epochs 50 --batch-size 16 --image-size 224 --balance balanced --best-metric accuracy
```

- **Automatic Pause & Resume**: Interrupting (`Ctrl+C`) saves `checkpoints/medfusionnet_last.pt`. Rerunning `python train.py` immediately resumes from the exact stopped epoch.
- **Early Stopping**: Monitored on validation accuracy/F1 to prevent overfitting.

### Evaluate Held-Out Test Set
```bash
python evaluate.py --checkpoint checkpoints/medfusionnet_best.pt
```

---

## 📄 Clinical Classes (HAM10000)

| Code | Disease Entity | Severity | Clinical Classification |
|---|---|---|---|
| `mel` | Melanoma | CRITICAL | Malignant Neoplasm |
| `bcc` | Basal Cell Carcinoma | HIGH | Malignant Non-Melanoma |
| `akiec`| Actinic Keratosis / Bowen's | ELEVATED | Premalignant Lesion |
| `bkl` | Benign Keratosis (Solar Lentigo) | LOW | Benign Epidermal Proliferation |
| `nv` | Melanocytic Nevus (Mole) | LOW | Common Benign Nevus |
| `vasc`| Vascular Lesion (Hemangioma) | LOW | Benign Vascular Formation |
| `df` | Dermatofibroma | LOW | Benign Histiocytoma |

---

*Academic clinical decision support prototype developed for dermoscopic lesion risk stratification.*
