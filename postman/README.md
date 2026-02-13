# PhytoSense Postman Testing Collections

This directory contains professional Postman collections for comprehensive testing of the PhytoSense application.

## Collections Available

### 1. Performance Testing Collection
**File:** `PhytoSense_Performance_Testing.postman_collection.json`

**Tests Included:**
- P1: Homepage Response Time (< 1000ms target)
- P2: QSAR Prediction Performance (< 10000ms target) 
- P3: Molecular Descriptor Calculation Performance (< 5000ms target)
- P4: 3D Molecular Visualization Performance (< 15000ms target)
- P5: Load Test (Multiple Concurrent Requests)
- P6: Performance Summary Report

**Automated Validation:**
- Response time benchmarks
- Performance scoring (Excellent/Good/Needs Improvement)
- Throughput analysis
- System stability under load

### 2. Integration Testing Collection  
**File:** `PhytoSense_Integration_Testing.postman_collection.json`

**Tests Included:**
- I1: SMILES to Molecular Descriptors Integration
- I2: SMILES to QSAR Prediction Integration
- I3: SMILES to 3D Molecular Visualization Integration
- I4: Multi-Compound QSAR Comparison Integration
- I5: End-to-End Workflow Integration
- I6: Error Handling Integration
- I7: Integration Test Summary Report

**Automated Validation:**
- Component interaction verification
- Data consistency across workflows
- Pipeline integrity testing
- Error handling robustness

## How to Use

### Prerequisites
1. **Start PhytoSense Application:**
   ```bash
   python flask_app.py
   ```
   (Application must be running on http://127.0.0.1:5000)

2. **Install Postman:**
   - Download from https://www.postman.com/
   - Or use Postman web version

### Import Collections

1. **Open Postman**
2. **Click "Import"** (top-left corner)
3. **Select "Upload Files"**
4. **Choose the JSON files:**
   - `PhytoSense_Performance_Testing.postman_collection.json`
   - `PhytoSense_Integration_Testing.postman_collection.json`
5. **Click "Import"**

### Run Collections

#### Option 1: Run Entire Collection
1. **Right-click on collection name**
2. **Select "Run collection"**
3. **Click "Run PhytoSense [Collection Name]"**
4. **View automated test results**

#### Option 2: Run Individual Tests
1. **Expand collection folder**
2. **Click on individual test**
3. **Click "Send"**
4. **View results in "Test Results" tab**

#### Option 3: Collection Runner (Recommended)
1. **Click "Runner" button** (top-left)
2. **Select collection**
3. **Configure settings:**
   - Iterations: 1 (or more for stress testing)
   - Delay: 1000ms between requests
4. **Click "Run"**

## Expected Results

### Performance Testing Results
- **Homepage:** < 100ms (Excellent), < 500ms (Good)
- **QSAR Predictions:** < 2000ms (Excellent), < 5000ms (Good)
- **3D Visualization:** < 10000ms (Excellent), < 15000ms (Good)
- **Load Test:** 100% success rate with 20 concurrent requests

### Integration Testing Results
- **All 7 integration tests:** PASSED
- **Cross-component data consistency:** Validated
- **Error handling:** Robust error responses
- **End-to-end workflow:** Complete pipeline functioning

## Automated Reporting

Both collections include **automated console reporting**:
- Performance metrics with color-coded results
- Integration test summaries
- Detailed logging for troubleshooting
- Pass/fail status for each test

### View Results
1. **Console Output:** View in Postman Console (bottom panel)
2. **Test Results:** Each request shows pass/fail status
3. **Collection Summary:** Final report after collection run

## Professional Features

### Performance Collection Features
- **Response time tracking** with millisecond precision
- **Performance scoring** (Excellent/Good/Needs Improvement)
- **Load testing** with concurrent request simulation
- **Comprehensive benchmarking** across all endpoints

### Integration Collection Features
- **Data flow validation** across multiple components
- **Cross-request data persistence** using global variables
- **Pipeline integrity testing** from SMILES to predictions
- **Error resilience testing** with invalid inputs

## Troubleshooting

### Common Issues
1. **Connection refused:** Ensure Flask app is running on port 5000
2. **Test failures:** Check Flask app logs for errors
3. **Timeout errors:** Increase timeout in individual requests if needed

### Debug Mode
- Each collection includes debug endpoints
- Console logging shows detailed execution flow
- Global variables track data between requests

## Academic Documentation

These collections provide **complete academic evidence** for:
- **Performance benchmarking** with quantitative metrics
- **Integration validation** with component interaction testing
- **System reliability** assessment under various conditions
- **Professional testing methodology** following industry standards

Perfect for academic submissions, technical documentation, and professional presentations.