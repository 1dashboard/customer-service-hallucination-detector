## Purpose

FastAPI REST API providing detection endpoints for the hallucination detection engine. Supports single reply detection, batch detection, file upload, historical result queries, evaluation, and health checks. Enables integration with external systems (ticketing, customer service backend).

## Requirements

### Requirement: Single reply detection endpoint
The system SHALL provide `POST /api/detect/single` that accepts a single reply item (user_question, system_reply, knowledge_base) and returns a detection result.

#### Scenario: Detect hallucination in single reply
- **WHEN** POST /api/detect/single with {"user_question": "...", "system_reply": "...", "knowledge_base": "..."}
- **THEN** system SHALL return {"id": "auto-generated", "is_hallucination": true/false, "detection_layer": "L1"|"L2"|"L3", "output_type": "...", "confidence": "HIGH"|"MEDIUM"|"LOW", "reason": "..."}

### Requirement: Batch detection endpoint
The system SHALL provide `POST /api/detect/batch` that accepts an array of reply items and returns detection results for all items. Each item MUST have user_question, system_reply, and knowledge_base fields. Items MAY include an optional id field.

#### Scenario: Batch detect multiple replies
- **WHEN** POST /api/detect/batch with an array of 20 reply objects
- **THEN** system SHALL return a JSON array of 20 detection results within 60 seconds

#### Scenario: Invalid input rejected
- **WHEN** POST /api/detect/batch with missing required fields
- **THEN** system SHALL return HTTP 422 with field-level error details

### Requirement: Upload and detect from file
The system SHALL provide `POST /api/detect/upload` that accepts a JSON file upload containing reply items and returns detection results.

#### Scenario: Upload JSON file for detection
- **WHEN** POST /api/detect/upload with multipart/form-data containing a valid JSON file
- **THEN** system SHALL parse the file, run detection, return results, and persist to database

### Requirement: Query historical results
The system SHALL provide `GET /api/results` that supports filtering by batch_id, output_type, is_hallucination, and date_range. Results SHALL be paginated (default: 20 per page).

#### Scenario: Filter results by hallucination type
- **WHEN** GET /api/results?output_type=capability_overreach&page=1&page_size=20
- **THEN** system SHALL return paginated results matching the filter

### Requirement: Evaluation endpoint
The system SHALL provide `POST /api/evaluate` that accepts a batch_id and ground truth data, computes evaluation metrics, and returns the report.

#### Scenario: Run evaluation against ground truth
- **WHEN** POST /api/evaluate with {"batch_id": "xxx", "ground_truth": [...]}
- **THEN** system SHALL return {"accuracy": 0.85, "precision": 0.90, "recall": 0.88, "f1": 0.89, "false_positives": [...], "false_negatives": [...]}

### Requirement: Health check endpoint
The system SHALL provide `GET /api/health` returning service status and LLM API availability.

#### Scenario: Service is healthy
- **WHEN** GET /api/health
- **THEN** system SHALL return {"status": "ok", "llm_available": true/false}

### Requirement: CORS support
The system SHALL enable CORS for all origins to allow Streamlit and other frontends to call the API.

#### Scenario: Cross-origin request succeeds
- **WHEN** an HTTP request from a different origin (e.g., streamlit on port 8501) calls the API
- **THEN** system SHALL respond with appropriate CORS headers
