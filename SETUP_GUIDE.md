# 🚀 Resume Screening Project — Complete Setup Guide

## Project Overview

| Component | Technology |
|---|---|
| Model 1 | TF-IDF + Random Forest |
| Model 2 | BERT (sentence-transformers) + Logistic Regression |
| ML Tracking | MLflow |
| Web UI | Streamlit |
| CI/CD | Jenkins |
| Testing | pytest |

---

## PART 1 — Python Environment Setup

### Step 1: Navigate to the project folder
Open Command Prompt and run:
```cmd
cd F:\resumescreenig\resumeScreening
```

### Step 2: Create a virtual environment
```cmd
python -m venv venv
```

### Step 3: Activate the virtual environment
```cmd
venv\Scripts\activate
```
You should see `(venv)` at the start of your prompt.

### Step 4: Install all dependencies
```cmd
pip install -r requirements.txt
```
> ⏳ This takes 5-10 minutes. PyTorch + sentence-transformers are large.

---

## PART 2 — Train the ML Models

### Step 5: Generate data + train both models
```cmd
python train.py
```

This will:
- Generate 400 synthetic resume-JD training samples
- Train **Model 1** (TF-IDF + Random Forest) → logged to MLflow
- Train **Model 2** (BERT Similarity) → logged to MLflow
- Save models to `saved_models/`
- Print a comparison table of both models

Expected output:
```
🚀 RESUME SCREENING - ML TRAINING PIPELINE
[MLflow] Tracking URI: mlruns
[Data] Generating synthetic dataset...

MODEL 1: TF-IDF + Random Forest
[Model 1] Training on 320 samples...
[Model 1] ✅ Accuracy: 0.8750 | F1: 0.8800 | AUC: 0.9200

MODEL 2: BERT Similarity + Logistic Regression
[Model 2] Loading BERT model: all-MiniLM-L6-v2...
[Model 2] ✅ Accuracy: 0.9000 | F1: 0.9100 | AUC: 0.9500
```

### Step 6: View MLflow dashboard
Open a **new** terminal and run:
```cmd
cd F:\resumescreenig\resumeScreening
venv\Scripts\activate
mlflow ui --port 5000
```
Then open in browser: **http://localhost:5000**

You'll see both experiments with metrics, parameters, and model artifacts.

---

## PART 3 — Run the Streamlit App

### Step 7: Launch the web UI
```cmd
streamlit run app/streamlit_app.py
```

Opens at: **http://localhost:8501**

Features:
- Upload resume (PDF, DOCX, TXT) or paste text
- Enter job description
- See both model scores side-by-side with gauge charts
- Adjust shortlist threshold in sidebar
- Download results as CSV

---

## PART 4 — Jenkins Setup (Step by Step)

### Step 8: Install Java (Jenkins requires Java 17+)

1. Go to: https://adoptium.net/
2. Download **OpenJDK 17 (LTS)** for Windows x64
3. Run the installer
4. After install, verify:
```cmd
java -version
```
Expected output: `openjdk version "17.x.x"`

### Step 9: Download Jenkins

1. Go to: https://www.jenkins.io/download/
2. Click **Windows** under LTS Release
3. Download `jenkins.msi` (~90 MB)

### Step 10: Install Jenkins

1. Double-click `jenkins.msi`
2. Click **Next** through the installer
3. **Installation folder**: Leave as default (`C:\Program Files\Jenkins`)
4. **Service Account**: Choose "Run service as LocalSystem" (easier for beginners)
5. **Port**: Leave as **8080**
6. **Java**: It should auto-detect Java 17
7. Click **Install**
8. Wait for installation to complete (~2-3 minutes)
9. Click **Finish**

### Step 11: First-time Jenkins setup

1. Open browser and go to: **http://localhost:8080**
2. You'll see "Unlock Jenkins"
3. Open the file shown on screen (usually):
```
C:\Program Files\Jenkins\secrets\initialAdminPassword
```
4. Copy the password and paste it into the browser
5. Click **Install suggested plugins** (recommended)
6. Wait for plugins to install (~5 minutes)
7. Create your **Admin account**:
   - Username: `admin`
   - Password: (choose one)
   - Full name: Your Name
   - Email: your@email.com
8. Click **Save and Continue**
9. Instance Configuration: Leave URL as `http://localhost:8080/` → click **Save and Finish**
10. Click **Start using Jenkins**

### Step 12: Install additional plugins

1. Go to **Manage Jenkins** → **Plugins** → **Available plugins**
2. Search and install these (tick all, then click Install):
   - **Pipeline** (already installed)
   - **Git plugin** (already installed)
   - **GitHub Integration Plugin**
   - **Pipeline: Stage View**
   - **Blue Ocean** (beautiful UI — optional but recommended)
   - **Workspace Cleanup Plugin**
3. Click **Restart Jenkins when installation is complete**

### Step 13: Set up Git (if you haven't already)

**Option A — Use a local Git repo** (simplest, no GitHub needed):
```cmd
cd F:\resumescreenig\resumeScreening
git init
git add .
git commit -m "Initial commit: Resume Screening Project"
```

**Option B — Use GitHub** (allows Jenkins to auto-trigger on push):
1. Create a new repo on GitHub.com
2. Push your code:
```cmd
cd F:\resumescreenig\resumeScreening
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/resumeScreening.git
git push -u origin main
```

### Step 14: Create the Jenkins Pipeline Job

1. On Jenkins dashboard, click **New Item**
2. Enter name: `resume-screening-pipeline`
3. Select **Pipeline** → click **OK**
4. In the configuration page:

**For local Git repo:**
- Under **Pipeline**, select "Pipeline script from SCM"
- **SCM**: Git
- **Repository URL**: `file:///F:/resumescreenig/resumeScreening`
- **Branch**: `*/main` or `*/master`
- **Script Path**: `Jenkinsfile`

**For GitHub:**
- Under **Pipeline**, select "Pipeline script from SCM"
- **SCM**: Git
- **Repository URL**: `https://github.com/YOUR_USERNAME/resumeScreening.git`
- **Branch**: `*/main`
- **Script Path**: `Jenkinsfile`
- (Add GitHub credentials if private repo)

5. Click **Save**

### Step 15: Run the Jenkins Pipeline

1. On the job page, click **Build Now**
2. Click on the build number (e.g., `#1`) to see progress
3. Click **Console Output** to see live logs

The pipeline will run 7 stages:
```
📥 Checkout → 🔧 Setup → 📊 Generate Data → 🤖 Train Models
    → 🧪 Test → 🚦 Quality Gate → 🚀 Deploy
```

### Step 16: Set up Auto-trigger on Code Changes

**For local polling (Jenkins polls every 5 minutes):**
Already configured in Jenkinsfile:
```groovy
triggers { pollSCM("H/5 * * * *") }
```

**For GitHub Webhooks (instant trigger on git push):**
1. In Jenkins, go to your job → **Configure**
2. Under **Build Triggers**, tick **GitHub hook trigger for GITScm polling**
3. In GitHub repo → **Settings** → **Webhooks** → **Add webhook**
4. Payload URL: `http://YOUR_IP:8080/github-webhook/`
   (Use ngrok to expose local Jenkins: `ngrok http 8080`)
5. Content type: `application/json`
6. Events: **Just the push event**
7. Click **Add webhook**

Now every `git push` triggers the pipeline automatically!

---

## PART 5 — Day-to-Day Workflow

### Making code changes + triggering Jenkins

```cmd
# Make your changes to models, data, etc.
# Then commit and push:
git add .
git commit -m "Improved Model 1 hyperparameters"
git push origin main
```

Jenkins auto-detects the push and:
1. Pulls your new code
2. Retrains both models
3. Logs new metrics to MLflow
4. Runs tests
5. Deploys updated models

---

## PART 6 — Run Tests Manually

```cmd
cd F:\resumescreenig\resumeScreening
venv\Scripts\activate

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_models.py::TestTFIDFRandomForestScreener -v

# Run with coverage
pytest tests/ -v --cov=models
```

---

## PART 7 — Quick Reference Commands

| Action | Command |
|---|---|
| Train models | `python train.py` |
| View MLflow | `mlflow ui --port 5000` |
| Launch UI | `streamlit run app/streamlit_app.py` |
| Screen resume (CLI) | `python predict.py --resume resume.pdf --jd jd.txt` |
| Interactive CLI | `python predict.py` |
| Run tests | `pytest tests/ -v` |
| Only train Model 1 | `python train.py --model1-only` |
| Only train Model 2 | `python train.py --model2-only` |
| Skip data generation | `python train.py --skip-generate` |

---

## Project Structure

```
resumeScreening/
├── data/
│   ├── generate_data.py       ← Synthetic data generator
│   └── resume_dataset.csv     ← Generated training data
├── models/
│   ├── __init__.py
│   ├── model1_tfidf_rf.py     ← TF-IDF + Random Forest (Model 1)
│   └── model2_bert.py         ← BERT + Logistic Regression (Model 2)
├── app/
│   └── streamlit_app.py       ← Web UI
├── tests/
│   └── test_models.py         ← pytest unit tests
├── saved_models/              ← Trained model files (.pkl)
├── production_models/         ← Jenkins-deployed models
├── mlruns/                    ← MLflow tracking data
├── train.py                   ← Main training script
├── predict.py                 ← CLI prediction tool
├── Jenkinsfile                ← CI/CD pipeline
├── requirements.txt           ← Python dependencies
└── SETUP_GUIDE.md             ← This file
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named X"**
→ Make sure venv is activated: `venv\Scripts\activate`
→ Reinstall: `pip install -r requirements.txt`

**BERT model download fails**
→ Check internet connection (downloads ~90MB on first run)
→ Or manually download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

**Jenkins port 8080 already in use**
→ Change port in Jenkins installer, or kill the process using port 8080

**Jenkins "java not found"**
→ Reinstall Java 17 and ensure it's added to PATH

**MLflow UI shows no experiments**
→ Run `python train.py` first to create experiments
→ Make sure you're running `mlflow ui` from the project directory

**Streamlit "Model not found"**
→ Run `python train.py` first to create `saved_models/`
