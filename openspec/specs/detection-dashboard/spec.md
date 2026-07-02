## Purpose

Streamlit web dashboard providing a visual interface for the hallucination detection system. Supports data upload, detection result browsing with filters, evaluation metrics visualization, misclassification analysis, and batch history navigation.

## Requirements

### Requirement: Data upload page
The system SHALL provide a page where users can upload a JSON file containing reply data for batch detection.

#### Scenario: User uploads reply data
- **WHEN** user uploads a valid JSON file via the file upload widget
- **THEN** system SHALL display a preview table of the uploaded data and a "Run Detection" button

#### Scenario: Invalid file rejected
- **WHEN** user uploads a file that is not valid JSON or has missing required fields
- **THEN** system SHALL display an error message explaining what is wrong

### Requirement: Detection results browser
The system SHALL display detection results in a filterable, sortable table showing: reply ID, user question (truncated), hallucination status (color-coded), output type, detection layer, confidence, and a detail expand button.

#### Scenario: View all detection results
- **WHEN** detection completes for a batch of 20 replies
- **THEN** system SHALL display a table with 20 rows, hallucination-positive rows highlighted in red, non-hallucination rows in green

#### Scenario: Filter results by type
- **WHEN** user selects "capability_overreach" from the type filter dropdown
- **THEN** system SHALL show only rows with output_type=capability_overreach

#### Scenario: Expand single result detail
- **WHEN** user clicks "View Detail" on a result row
- **THEN** system SHALL expand to show: full user question, full system reply, full knowledge base reference, detection reason, and confidence score

### Requirement: Evaluation metrics dashboard
The system SHALL provide a page displaying evaluation metrics (accuracy, precision, recall, F1) as metric cards, and lists of false positives and false negatives with case details.

#### Scenario: View evaluation results
- **WHEN** user navigates to the evaluation page after uploading ground truth
- **THEN** system SHALL display four metric cards (Accuracy, Precision, Recall, F1) and two tables (False Positives, False Negatives)

### Requirement: Misclassification analysis page
The system SHALL provide an analysis page that groups misclassified cases by error type and provides potential causes.

#### Scenario: View misclassification breakdown
- **WHEN** user views the analysis page
- **THEN** system SHALL show groups of false positives and false negatives with common patterns identified

### Requirement: Batch history sidebar
The system SHALL display a sidebar listing past detection batches with timestamp, filename, and hallucination count, allowing users to switch between historical results.

#### Scenario: Switch to a previous batch
- **WHEN** user clicks a past batch in the sidebar
- **THEN** system SHALL load and display that batch's detection results

### Requirement: Export results
The system SHALL allow users to export detection results as JSON or CSV files.

#### Scenario: Export as JSON
- **WHEN** user clicks "Export JSON" on the results page
- **THEN** system SHALL download a JSON file containing all detection results for the current batch

#### Scenario: Export as CSV
- **WHEN** user clicks "Export CSV" on the results page
- **THEN** system SHALL download a CSV file with columns: id, is_hallucination, output_type, detection_layer, confidence, reason
