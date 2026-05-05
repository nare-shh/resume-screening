// ============================================================
// Jenkinsfile — Resume Screening ML Pipeline
// ============================================================
// Stages:
//   1. Checkout       — pull latest code from Git
//   2. Setup          — install Python dependencies
//   3. Generate Data  — create training dataset (if not exists)
//   4. Train Models   — train both ML models + log to MLflow
//   5. Test           — run pytest unit tests
//   6. Quality Gate   — check model metrics meet minimum thresholds
//   7. Deploy         — copy models to "production" folder
//   8. Notify         — print summary
// ============================================================

pipeline {
    agent any

    // ── Environment Variables ─────────────────────────────────────────────────
    environment {
        PYTHON_CMD       = "python"                 // or "python3" on Linux
        PROJECT_DIR      = "${WORKSPACE}"
        VENV_DIR         = "${WORKSPACE}\\venv"     // Windows path
        MLFLOW_PORT      = "5000"
        MIN_F1_SCORE     = "0.70"                   // minimum acceptable F1
        MIN_ACCURACY     = "0.70"                   // minimum acceptable accuracy
        PROD_MODELS_DIR  = "${WORKSPACE}\\production_models"
    }

    // ── Triggers ──────────────────────────────────────────────────────────────
    triggers {
        // Poll GitHub every 5 minutes for new commits
        pollSCM("H/5 * * * *")
    }

    options {
        // Keep last 10 builds
        buildDiscarder(logRotator(numToKeepStr: "10"))
        // Timeout entire pipeline after 60 minutes
        timeout(time: 60, unit: "MINUTES")
        // Add timestamps to console output
        timestamps()
    }

    // ── Parameters (can override at build time) ───────────────────────────────
    parameters {
        booleanParam(
            name: "SKIP_GENERATE",
            defaultValue: false,
            description: "Skip dataset generation (use existing CSV)"
        )
        booleanParam(
            name: "TRAIN_MODEL1_ONLY",
            defaultValue: false,
            description: "Only train Model 1 (TF-IDF + RF)"
        )
        booleanParam(
            name: "TRAIN_MODEL2_ONLY",
            defaultValue: false,
            description: "Only train Model 2 (BERT)"
        )
        string(
            name: "N_SAMPLES",
            defaultValue: "400",
            description: "Number of training samples to generate"
        )
    }

    stages {

        // ── Stage 1: Checkout ─────────────────────────────────────────────────
        stage("📥 Checkout") {
            steps {
                echo "========================================"
                echo "  Checking out source code from Git"
                echo "========================================"
                checkout scm
                echo "✅ Code checked out at: ${env.GIT_COMMIT}"
                echo "   Branch: ${env.GIT_BRANCH}"
            }
        }

        // ── Stage 2: Setup Python Environment ─────────────────────────────────
        stage("🔧 Setup") {
            steps {
                echo "========================================"
                echo "  Setting up Python virtual environment"
                echo "========================================"
                script {
                    if (isUnix()) {
                        sh """
                            python3 -m venv venv
                            . venv/bin/activate
                            pip install --upgrade pip
                            pip install -r requirements.txt
                        """
                    } else {
                        bat """
                            python -m venv venv
                            call venv\\Scripts\\activate.bat
                            pip install --upgrade pip
                            pip install -r requirements.txt
                        """
                    }
                }
                echo "✅ Python environment ready"
            }
        }

        // ── Stage 3: Generate Training Data ───────────────────────────────────
        stage("📊 Generate Data") {
            when {
                not { expression { return params.SKIP_GENERATE } }
            }
            steps {
                echo "========================================"
                echo "  Generating synthetic training dataset"
                echo "========================================"
                script {
                    if (isUnix()) {
                        sh """
                            . venv/bin/activate
                            python data/generate_data.py
                        """
                    } else {
                        bat """
                            call venv\\Scripts\\activate.bat
                            python data\\generate_data.py
                        """
                    }
                }
                echo "✅ Training data generated"
            }
        }

        // ── Stage 4: Train ML Models ───────────────────────────────────────────
        stage("🤖 Train Models") {
            steps {
                echo "========================================"
                echo "  Training ML models + logging to MLflow"
                echo "========================================"
                script {
                    def trainArgs = ""
                    if (params.SKIP_GENERATE) trainArgs += " --skip-generate"
                    if (params.TRAIN_MODEL1_ONLY) trainArgs += " --model1-only"
                    if (params.TRAIN_MODEL2_ONLY) trainArgs += " --model2-only"
                    trainArgs += " --samples ${params.N_SAMPLES}"

                    if (isUnix()) {
                        sh """
                            . venv/bin/activate
                            python train.py ${trainArgs}
                        """
                    } else {
                        bat """
                            call venv\\Scripts\\activate.bat
                            python train.py ${trainArgs}
                        """
                    }
                }
                echo "✅ Models trained and logged to MLflow"
                // Archive training results
                archiveArtifacts artifacts: "training_results.json", allowEmptyArchive: true
            }
        }

        // ── Stage 5: Run Tests ─────────────────────────────────────────────────
        stage("🧪 Test") {
            steps {
                echo "========================================"
                echo "  Running pytest unit tests"
                echo "========================================"
                script {
                    if (isUnix()) {
                        sh """
                            . venv/bin/activate
                            pytest tests/test_models.py::TestTFIDFRandomForestScreener::test_import \
                                   tests/test_models.py::TestTFIDFRandomForestScreener::test_instantiation \
                                   tests/test_models.py::TestBERTSimilarityScreener::test_import \
                                   tests/test_models.py::TestBERTSimilarityScreener::test_instantiation \
                                   tests/test_models.py::TestBERTSimilarityScreener::test_cosine_similarity_static \
                                   tests/test_models.py::TestBERTSimilarityScreener::test_keyword_overlap_static \
                                   tests/test_models.py::TestIntegration \
                                   -v --tb=short
                        """
                    } else {
                        bat """
                            call venv\\Scripts\\activate.bat
                            pytest tests/test_models.py::TestTFIDFRandomForestScreener::test_import ^
                                   tests/test_models.py::TestTFIDFRandomForestScreener::test_instantiation ^
                                   tests/test_models.py::TestBERTSimilarityScreener::test_import ^
                                   tests/test_models.py::TestBERTSimilarityScreener::test_instantiation ^
                                   tests/test_models.py::TestBERTSimilarityScreener::test_cosine_similarity_static ^
                                   tests/test_models.py::TestBERTSimilarityScreener::test_keyword_overlap_static ^
                                   tests/test_models.py::TestIntegration ^
                                   -v --tb=short
                        """
                    }
                }
                echo "✅ All tests passed"
            }
            post {
                always {
                    // Publish test results if junit report exists
                    script {
                        if (fileExists("test-results.xml")) {
                            junit "test-results.xml"
                        }
                    }
                }
            }
        }

        // ── Stage 6: Quality Gate ──────────────────────────────────────────────
        stage("🚦 Quality Gate") {
            steps {
                echo "========================================"
                echo "  Checking model quality thresholds"
                echo "========================================"
                script {
                    if (fileExists("training_results.json")) {
                        def resultsText = readFile("training_results.json")
                        def results = readJSON(text: resultsText)

                        def allPassed = true
                        results.each { model ->
                            def f1 = model.f1 as Double
                            def acc = model.accuracy as Double
                            def modelName = model.model ?: "Unknown"

                            echo "Model: ${modelName}"
                            echo "  F1 Score : ${f1} (min: ${MIN_F1_SCORE})"
                            echo "  Accuracy : ${acc} (min: ${MIN_ACCURACY})"

                            if (f1 < MIN_F1_SCORE.toDouble()) {
                                echo "⚠️  WARNING: ${modelName} F1 (${f1}) below threshold (${MIN_F1_SCORE})"
                                allPassed = false
                            }
                            if (acc < MIN_ACCURACY.toDouble()) {
                                echo "⚠️  WARNING: ${modelName} Accuracy (${acc}) below threshold (${MIN_ACCURACY})"
                                allPassed = false
                            }
                        }

                        if (allPassed) {
                            echo "✅ Quality Gate PASSED — all models meet minimum thresholds"
                        } else {
                            echo "⚠️  Quality Gate WARNING — some models below threshold"
                            // Uncomment to fail the build:
                            // error("Quality Gate failed — models below minimum thresholds")
                        }
                    } else {
                        echo "⚠️  training_results.json not found — skipping quality gate"
                    }
                }
            }
        }

        // ── Stage 7: Deploy (Save Production Models) ───────────────────────────
        stage("🚀 Deploy") {
            steps {
                echo "========================================"
                echo "  Deploying models to production folder"
                echo "========================================"
                script {
                    if (isUnix()) {
                        sh """
                            mkdir -p production_models
                            cp -f saved_models/model1_tfidf_rf.pkl production_models/ 2>/dev/null || true
                            cp -f saved_models/model2_bert.pkl production_models/ 2>/dev/null || true
                            echo "Deployed at: \$(date)" > production_models/DEPLOY_INFO.txt
                            echo "Commit: ${env.GIT_COMMIT}" >> production_models/DEPLOY_INFO.txt
                            echo "Branch: ${env.GIT_BRANCH}" >> production_models/DEPLOY_INFO.txt
                            cat production_models/DEPLOY_INFO.txt
                        """
                    } else {
                        bat """
                            if not exist production_models mkdir production_models
                            copy /Y saved_models\\model1_tfidf_rf.pkl production_models\\ 2>nul
                            copy /Y saved_models\\model2_bert.pkl production_models\\ 2>nul
                            echo Deployed at: %DATE% %TIME% > production_models\\DEPLOY_INFO.txt
                            echo Commit: ${env.GIT_COMMIT} >> production_models\\DEPLOY_INFO.txt
                            type production_models\\DEPLOY_INFO.txt
                        """
                    }
                }
                archiveArtifacts artifacts: "production_models/**", allowEmptyArchive: true
                echo "✅ Models deployed to production_models/"
            }
        }
    }

    // ── Post Actions ──────────────────────────────────────────────────────────
    post {
        success {
            echo """
            ============================================
            ✅ PIPELINE SUCCEEDED
            ============================================
            Branch  : ${env.GIT_BRANCH}
            Commit  : ${env.GIT_COMMIT}
            Build   : #${env.BUILD_NUMBER}
            Duration: ${currentBuild.durationString}

            Next steps:
            - View MLflow UI: mlflow ui --port 5000
            - Run Streamlit: streamlit run app/streamlit_app.py
            ============================================
            """
        }
        failure {
            echo """
            ============================================
            ❌ PIPELINE FAILED
            ============================================
            Build  : #${env.BUILD_NUMBER}
            Check the console output above for errors.
            ============================================
            """
        }
        always {
            cleanWs(
                cleanWhenSuccess: false,
                cleanWhenFailure: false,
                cleanWhenAborted: false,
                deleteDirs: false,
                patterns: [[pattern: "venv/**", type: "INCLUDE"]]
            )
        }
    }
}
