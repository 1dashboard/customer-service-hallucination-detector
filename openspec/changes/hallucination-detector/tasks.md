## 1. Project Setup

- [x] 1.1 Create hallucination_detector/ directory structure (engine/, api/, dashboard/, data/, tests/)
- [x] 1.2 Create requirements.txt with dependencies: fastapi, uvicorn, streamlit, jieba, openai, pandas, python-dotenv
- [x] 1.3 Create .env.example with DEEPSEEK_API_KEY placeholder and .gitignore entry

## 2. Data Layer

- [x] 2.1 Define data models (DetectionInput, DetectionResult, BatchRecord) in engine/models.py
- [x] 2.2 Implement SQLite database initialization with tables: detection_results, detection_batches, evaluation_runs in engine/db.py
- [x] 2.3 Copy replies.json and ground_truth.json to data/ directory

## 3. Stage 1 Rule Engine (L1 + L2)

- [x] 3.1 Implement entity extractor (numeric + named entities) using jieba + regex in engine/rules.py
- [x] 3.2 Implement L1 direct contradiction detector (KB entity vs reply entity comparison)
- [x] 3.3 Implement L2 capability overreach detector (KB negation pattern matching)
- [x] 3.4 Implement classification mapper (L1/L2/L3 → output layer business labels) in engine/classifier.py
- [x] 3.5 Write unit tests for L1 and L2 rules against known cases in tests/test_rules.py

## 4. Stage 2 LLM Judge (L3)

- [x] 4.1 Implement DeepSeek API client (OpenAI SDK compatible) with retry logic in engine/llm_judge.py
- [x] 4.2 Design and implement the hallucination detection prompt (system + user template) returning structured JSON
- [x] 4.3 Implement fallback handling (null result on persistent API failure)
- [x] 4.4 Write unit tests for LLM judge with mock responses in tests/test_llm.py

## 5. Detection Pipeline

- [x] 5.1 Implement pipeline orchestrator (Stage 1 → HIGH confidence output, LOW → Stage 2) in engine/pipeline.py
- [x] 5.2 Run full pipeline on 20 replies and save results to JSON
- [x] 5.3 Write integration test: run pipeline on test fixtures and verify output schema

## 6. FastAPI Backend

- [x] 6.1 Create FastAPI app with CORS and health check endpoint in api/main.py
- [x] 6.2 Implement POST /api/detect/single endpoint
- [x] 6.3 Implement POST /api/detect/batch endpoint
- [x] 6.4 Implement POST /api/detect/upload endpoint (file upload)
- [x] 6.5 Implement GET /api/results endpoint with pagination and filters
- [x] 6.6 Implement POST /api/evaluate endpoint
- [x] 6.7 Write API integration tests in tests/test_api.py

## 7. Streamlit Dashboard

- [x] 7.1 Create Streamlit app entry point in dashboard/app.py with sidebar navigation
- [x] 7.2 Implement upload page (file upload → API call → results display)
- [x] 7.3 Implement results browser page (filterable table + detail expansion)
- [x] 7.4 Implement evaluation metrics page (accuracy/precision/recall/F1 cards + FP/FN tables)
- [x] 7.5 Implement misclassification analysis page
- [x] 7.6 Implement batch history sidebar and export (JSON/CSV) functionality

## 8. Evaluation Against Ground Truth

- [x] 8.1 Implement evaluation engine (compare results vs ground_truth, compute metrics) in engine/evaluator.py
- [x] 8.2 Run evaluation on 20-sample dataset and generate report
- [x] 8.3 Analyze misclassified cases and document error patterns

## 9. Run Script & Documentation

- [x] 9.1 Create run.py one-click startup script (launch both FastAPI and Streamlit)
- [x] 9.2 Write README.md: classification system overview, detection method, detection rate data, AI tool usage, setup instructions
