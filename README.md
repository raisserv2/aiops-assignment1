# AIOps Assignment 1

**Myself:** Mohammed Khaja Ghouse Mohiuddin 
**Partner (Q4):** Mudda Manikanta Pruthvi Raj  
**Repo:** `raisserv2/aiops-assignment1`

---

## Repository Structure

```
.
├── README.md                  <- You are here
├── AI_DISCLOSURE.md           <- AI tool usage declaration
├── report.pdf                 <- 1-page write-up (all 4 questions)
├── report.tex                 <- LaTeX source for the report
├── train.py                   <- MLP training script with MLflow logging
├── environment.yml            <- Conda environment for reproducibility
├── mnist_data.csv.dvc         <- DVC pointer to the MNIST dataset
├── .dvc/config                <- DVC remote configuration (SSH)
├── screenshots/
│   ├── run_comparison.png     <- Q2: MLflow 6-run comparison view
│   └── rollback_proof.png     <- Q3: DVC rollback terminal output
└── mlruns/                    <- MLflow artifact store
```

---

## Question-wise Deliverables

### Q1 - Technical Debt Diagnosis (10 marks)
- **Where:** `report.pdf`, Section 1
- Identifies entanglement (CACE), undeclared consumers, and configuration/glue-code debt from the three scenarios, with a mitigation proposal using MLflow Projects.

### Q2 - MLflow Experiment Comparison (15 marks)
- **Comparison screenshot:** `screenshots/run_comparison.png`
- **Written analysis:** `report.pdf`, Section 2
- **Code:** `train.py` (see `mlflow.log_param()` and `mlflow.log_metric()` calls)
- 6 unique runs varying learning rate (0.001, 0.01, 0.0001) and hidden layer sizes (64,32 / 128,64 / 256,128). Best run: lr=0.001, hidden=256,128 at 97.51% accuracy.

### Q3 - DVC Data Versioning & Rollback (10 marks)
- **Rollback proof:** `screenshots/rollback_proof.png`
- v1: 1800 rows (wc -l shows 1801 with header), v2: 2801 rows (wc -l shows 2802)
- DVC remote: SSH (`ssh://khaja@10.252.48.235/home/khaja/dvc_remote`)
- Rollback via `git checkout v1` + `dvc checkout` confirmed row count matches v1.

### Q4 - End-to-End Reproducibility Drill (15 marks)
- **Partner A (Khaja):** Trained MLP on MNIST, logged run with params/metrics/seed/git_commit tag, DVC-versioned dataset, registered model as "my-classifier" and transitioned to Staging. Commit: `80d4b50`.
- **Partner B (Pruthvi Raj):** Cloned repo, reproduced the run, got accuracy 0.9784 (Partner A: 0.9782, diff = 0.0002, within tolerance). Logged reproducibility note in MLflow. See his commit on this repo.
- Both partners' contributions are distinguishable by commit author.

---

## How to Set Up and Run

### Prerequisites
- Ubuntu (tested on 24.04)
- Conda or Miniconda installed

### Steps

```bash
# 1. Clone and checkout
git clone https://github.com/raisserv2/aiops-assignment1.git
cd aiops-assignment1

# 2. Create environment
conda env create -f environment.yml
conda activate aiops-m1

# 3. Restore DVC-tracked data
dvc pull   # requires SSH access to the DVC remote

# 4. Start MLflow server (Terminal 1)
mlflow server --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# 5. Run training (Terminal 2)
python train.py --lr 0.001 --hidden "256,128" --epochs 20 --batch_size 200 --seed 42

# Expected output: Final accuracy ~ 0.9782
```

---

## Demo Video

[Link to demo video](https://drive.google.com/file/d/19PUov1A76tkPlwJKJnNMJAPfDAGp_iOK/view?usp=sharing)
