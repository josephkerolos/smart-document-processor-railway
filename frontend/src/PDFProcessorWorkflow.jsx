import React, { useState, useEffect, useRef } from 'react';
import './PDFProcessorWorkflow.css';

const PDFProcessorWorkflow = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedSchema, setSelectedSchema] = useState('941-X');
  const [expectedValue, setExpectedValue] = useState('');
  const [targetSizeMb, setTargetSizeMb] = useState('');
  const [processing, setProcessing] = useState(false);
  const [workflowSteps, setWorkflowSteps] = useState([]);
  const [currentResults, setCurrentResults] = useState(null);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [extractionProgress, setExtractionProgress] = useState({});
  const [compressionProgress, setCompressionProgress] = useState({});
  const wsRef = useRef(null);
  const fileInputRef = useRef(null);

  // Available schemas
  const schemas = [
    { value: '941-X', label: 'IRS Form 941-X' },
    { value: '941', label: 'IRS Form 941' },
    { value: '1040', label: 'IRS Form 1040' },
    { value: '2848', label: 'IRS Form 2848' },
    { value: '8821', label: 'IRS Form 8821' },
    { value: 'payroll', label: 'Payroll Data' },
    { value: 'generic', label: 'Generic Document' }
  ];

  // WebSocket connection management
  useEffect(() => {
    if (sessionId && !wsRef.current) {
      const wsUrl = `ws://localhost:4830/ws/enhanced/${sessionId}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error occurred');
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        wsRef.current = null;
      };
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [sessionId]);

  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'workflow_plan':
        setWorkflowSteps(message.data.steps);
        break;
      
      case 'processing_plan':
        // Update steps if using old format
        if (message.data.steps) {
          setWorkflowSteps(message.data.steps);
        }
        break;
      
      case 'time_update':
        setElapsedTime(message.data.elapsed_seconds);
        break;
      
      case 'cleaning_complete':
        updateResultsSection('cleaning', message.data);
        break;
      
      case 'splitting_complete':
        updateResultsSection('splitting', message.data);
        break;
      
      case 'extraction_started':
        // Initialize extraction progress for all documents
        const initialProgress = {};
        message.data.documents.forEach((doc, idx) => {
          initialProgress[idx] = {
            filename: doc.filename,
            status: 'pending',
            extraction_time: null
          };
        });
        setExtractionProgress(initialProgress);
        updateResultsSection('extraction_progress', message.data);
        break;
        
      case 'document_extraction_started':
        // Update specific document status
        setExtractionProgress(prev => ({
          ...prev,
          [message.data.index]: {
            ...prev[message.data.index],
            status: 'extracting',
            message: message.data.message
          }
        }));
        break;
        
      case 'document_extraction_completed':
        // Update specific document completion
        setExtractionProgress(prev => ({
          ...prev,
          [message.data.index]: {
            ...prev[message.data.index],
            status: message.data.status,
            extraction_time: message.data.extraction_time,
            has_errors: message.data.has_errors,
            pages_processed: message.data.pages_processed,
            message: message.data.message
          }
        }));
        break;
        
      case 'extraction_progress':
        updateResultsSection('extraction_progress', message.data);
        break;
      
      case 'compression_started':
        // Initialize compression progress for all documents
        const initialCompressionProgress = {};
        message.data.documents.forEach((doc, idx) => {
          initialCompressionProgress[idx] = {
            filename: doc.filename,
            status: 'pending',
            compression_time: null,
            original_size_mb: null,
            compressed_size_mb: null,
            compression_ratio: null
          };
        });
        setCompressionProgress(initialCompressionProgress);
        updateResultsSection('compression_progress', message.data);
        break;
        
      case 'document_compression_started':
        // Update specific document compression status
        setCompressionProgress(prev => ({
          ...prev,
          [message.data.index]: {
            ...prev[message.data.index],
            status: 'compressing'
          }
        }));
        break;
        
      case 'document_compression_completed':
        // Update specific document compression completion
        setCompressionProgress(prev => ({
          ...prev,
          [message.data.index]: {
            ...prev[message.data.index],
            status: message.data.status,
            compression_time: message.data.compression_time,
            original_size_mb: message.data.original_size_mb,
            compressed_size_mb: message.data.compressed_size_mb,
            compression_ratio: message.data.compression_ratio
          }
        }));
        break;
      
      case 'organization_complete':
        updateResultsSection('organization', message.data);
        break;
      
      case 'google_drive_complete':
        updateResultsSection('google_drive', message.data);
        break;
      
      case 'workflow_complete':
        setProcessing(false);
        setCurrentResults(message.data);
        break;
      
      case 'error':
        setError(message.data.message);
        setProcessing(false);
        break;
      
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const updateResultsSection = (section, data) => {
    setCurrentResults(prev => ({
      ...prev,
      [section]: data
    }));
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setError(null);
    setCurrentResults(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    setSelectedFiles(files);
    setError(null);
    setCurrentResults(null);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const startProcessing = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one file');
      return;
    }

    setProcessing(true);
    setError(null);
    setCurrentResults(null);
    setWorkflowSteps([]);
    setElapsedTime(0);
    setExtractionProgress({});
    setCompressionProgress({});

    const formData = new FormData();
    formData.append('file', selectedFiles[0]); // Enhanced workflow currently handles one file
    formData.append('selected_schema', selectedSchema);
    if (expectedValue) {
      formData.append('expected_value', expectedValue);
    }
    if (targetSizeMb) {
      formData.append('target_size_mb', targetSizeMb);
    }

    try {
      const response = await fetch('http://localhost:4830/api/process-enhanced', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (response.ok) {
        setSessionId(result.session_id);
        // The workflow results are already in the response
        if (result.success) {
          setProcessing(false);
          setCurrentResults(result);
          // Update final step
          setWorkflowSteps(prev => prev.map(step => 
            step.id === 'complete' ? {...step, status: 'completed'} : step
          ));
        }
      } else {
        setError(result.error || 'Processing failed');
        setProcessing(false);
      }
    } catch (err) {
      setError(`Failed to start processing: ${err.message}`);
      setProcessing(false);
    }
  };

  const downloadArchive = (archiveName) => {
    if (!currentResults || !sessionId) return;
    
    const downloadUrl = `http://localhost:4830/api/download-archive/${sessionId}/${archiveName}`;
    window.open(downloadUrl, '_blank');
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStepIcon = (status) => {
    switch (status) {
      case 'completed': return '✅';
      case 'in_progress': return '⏳';
      case 'failed': return '❌';
      default: return '⚪';
    }
  };

  return (
    <div className="pdf-processor-workflow">
      <div className="workflow-header">
        <h1>Enhanced Document Processing Workflow</h1>
        <p className="workflow-description">
          Process documents with automatic cleaning, splitting, extraction, and organization
        </p>
      </div>

      {/* File Upload Section */}
      <div className="upload-section">
        <div 
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          {selectedFiles.length > 0 ? (
            <div className="selected-files">
              <h3>Selected Files:</h3>
              {selectedFiles.map((file, idx) => (
                <div key={idx} className="file-item">
                  📄 {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </div>
              ))}
            </div>
          ) : (
            <div className="drop-instructions">
              <span className="drop-icon">📤</span>
              <p>Drop PDF files here or click to browse</p>
            </div>
          )}
        </div>
      </div>

      {/* Configuration Section */}
      <div className="config-section">
        <div className="config-row">
          <div className="config-item">
            <label htmlFor="schema-select">Document Type:</label>
            <select
              id="schema-select"
              value={selectedSchema}
              onChange={(e) => setSelectedSchema(e.target.value)}
              disabled={processing}
            >
              {schemas.map(schema => (
                <option key={schema.value} value={schema.value}>
                  {schema.label}
                </option>
              ))}
            </select>
          </div>
          
          {selectedSchema === '941-X' && (
            <div className="config-item">
              <label htmlFor="expected-value">Expected Value (optional):</label>
              <input
                id="expected-value"
                type="text"
                placeholder="e.g., 38808.50"
                value={expectedValue}
                onChange={(e) => setExpectedValue(e.target.value)}
                disabled={processing}
              />
            </div>
          )}
          
          <div className="config-item">
            <label htmlFor="target-size">
              Target Total Size (MB) - Optional:
              <span className="tooltip" title="Leave empty to use default (9.0 MB). The system will divide this target by the number of documents to set individual compression targets. Files already within target won't be compressed."> ℹ️</span>
            </label>
            <input
              id="target-size"
              type="number"
              min="1"
              max="50"
              step="1"
              placeholder="9.0"
              value={targetSizeMb}
              onChange={(e) => setTargetSizeMb(e.target.value)}
              disabled={processing}
            />
          </div>
        </div>
        
        <button
          className="process-button"
          onClick={startProcessing}
          disabled={processing || selectedFiles.length === 0}
        >
          {processing ? 'Processing...' : 'Start Processing'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}

      {/* Workflow Progress */}
      {workflowSteps.length > 0 && (
        <div className="workflow-progress">
          <div className="progress-header">
            <h2>Processing Progress</h2>
            {processing && (
              <span className="elapsed-time">
                Elapsed: {formatTime(elapsedTime)}
              </span>
            )}
          </div>
          
          <div className="workflow-steps">
            {workflowSteps.map((step, idx) => (
              <div key={step.id} className={`workflow-step ${step.status}`}>
                <span className="step-icon">{getStepIcon(step.status)}</span>
                <span className="step-name">{step.name}</span>
                {step.message && (
                  <span className="step-message">{step.message}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results Section */}
      {currentResults && (
        <div className="results-section">
          <h2>Processing Results</h2>
          
          {/* Cleaning Results */}
          {currentResults.cleaning && (
            <div className="result-card">
              <h3>🧹 Document Cleaning</h3>
              <p>Total pages analyzed: {currentResults.cleaning.total_pages}</p>
              <p>Instruction pages removed: {currentResults.cleaning.removed_pages}</p>
              <p>Content pages kept: {currentResults.cleaning.kept_pages || (currentResults.cleaning.total_pages - currentResults.cleaning.removed_pages)}</p>
              {currentResults.cleaning.removed_pages > 0 && (
                <p className="cleaning-note">✅ Files will be marked as "cleaned_" in the output</p>
              )}
              {currentResults.cleaning.removed_pages === 0 && (
                <p className="cleaning-note">ℹ️ No instruction pages found - document appears clean</p>
              )}
            </div>
          )}
          
          {/* Splitting Results */}
          {currentResults.splitting && (
            <div className="result-card">
              <h3>✂️ Document Splitting</h3>
              <p>Split into {currentResults.splitting.document_count} individual documents:</p>
              <ul className="document-list">
                {currentResults.splitting.documents.map((doc, idx) => (
                  <li key={idx}>{doc.filename}</li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Parallel Extraction Progress */}
          {Object.keys(extractionProgress).length > 0 && (
            <div className="result-card">
              <h3>📊 Parallel Extraction Progress</h3>
              <div className="extraction-progress-grid">
                {Object.entries(extractionProgress).map(([idx, progress]) => (
                  <div key={idx} className={`extraction-item ${progress.status}`}>
                    <div className="extraction-filename">{progress.filename}</div>
                    <div className="extraction-status">
                      {progress.status === 'pending' && '⏳ Waiting...'}
                      {progress.status === 'extracting' && (
                        <>
                          🔄 Extracting...
                          {progress.message && <span style={{ fontSize: '11px', display: 'block' }}>{progress.message}</span>}
                        </>
                      )}
                      {progress.status === 'completed' && (
                        <>
                          ✅ Done ({progress.extraction_time?.toFixed(1)}s)
                          {progress.pages_processed && (
                            <span style={{ fontSize: '11px', display: 'block' }}>
                              {progress.pages_processed} pages processed
                            </span>
                          )}
                        </>
                      )}
                      {progress.status === 'failed' && '❌ Failed'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Parallel Compression Progress */}
          {Object.keys(compressionProgress).length > 0 && (
            <div className="result-card">
              <h3>🗜️ Parallel Compression Progress</h3>
              <div className="extraction-progress-grid">
                {Object.entries(compressionProgress).map(([idx, progress]) => (
                  <div key={idx} className={`extraction-item ${progress.status}`}>
                    <div className="extraction-filename">{progress.filename}</div>
                    <div className="extraction-status">
                      {progress.status === 'pending' && '⏳ Waiting...'}
                      {progress.status === 'compressing' && '🔄 Compressing...'}
                      {progress.status === 'completed' && (
                        <>
                          ✅ Done ({progress.compression_time?.toFixed(1)}s)
                          <span style={{ fontSize: '12px', display: 'block', marginTop: '2px' }}>
                            {progress.original_size_mb}MB → {progress.compressed_size_mb}MB 
                            ({progress.compression_ratio}% reduction)
                          </span>
                        </>
                      )}
                      {progress.status === 'failed' && '❌ Failed'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Organization Results */}
          {currentResults.organization && (
            <div className="result-card">
              <h3>📁 Document Organization</h3>
              <p>Organized into {currentResults.organization.quarters.length} quarters:</p>
              <ul className="quarter-list">
                {currentResults.organization.quarters.map((quarter, idx) => (
                  <li key={idx}>{quarter}</li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Google Drive Upload Results */}
          {currentResults.google_drive_upload && currentResults.google_drive_upload.success && (
            <div className="result-card" style={{ backgroundColor: '#e8f5e9', border: '2px solid #4caf50' }}>
              <h3>☁️ Google Drive Upload</h3>
              <p><strong>Status:</strong> ✅ Successfully uploaded to Google Drive</p>
              <p><strong>Folder Path:</strong> {currentResults.google_drive_upload.folder_path}</p>
              <div style={{ marginTop: '15px' }}>
                <a 
                  href={currentResults.google_drive_upload.google_drive_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: 'inline-block',
                    padding: '10px 20px',
                    backgroundColor: '#4caf50',
                    color: 'white',
                    textDecoration: 'none',
                    borderRadius: '4px',
                    fontSize: '16px',
                    fontWeight: '500'
                  }}
                >
                  📁 View in Google Drive
                </a>
              </div>
              <div style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
                <p><strong>Uploaded Files:</strong></p>
                <ul style={{ marginLeft: '20px', marginTop: '5px' }}>
                  <li>{currentResults.google_drive_upload.uploads.master_archive.name} ({(currentResults.google_drive_upload.uploads.master_archive.size / (1024 * 1024)).toFixed(2)} MB)</li>
                  <li>{currentResults.google_drive_upload.uploads.summary.name}</li>
                </ul>
              </div>
            </div>
          )}
          
          {/* Processing Stats */}
          {currentResults.processing_stats && (
            <div className="result-card">
              <h3>⏱️ Processing Statistics</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                <div>
                  <strong>Total Time:</strong> {currentResults.total_elapsed_time}s ({formatTime(Math.floor(currentResults.total_elapsed_time))})
                </div>
                <div>
                  <strong>Documents Processed:</strong> {currentResults.processing_stats.documents_processed}
                </div>
                <div>
                  <strong>Pages Removed:</strong> {currentResults.processing_stats.pages_removed}
                </div>
                <div>
                  <strong>Compressions Successful:</strong> {currentResults.processing_stats.compressions_successful}
                </div>
                <div>
                  <strong>Avg Time per Document:</strong> {currentResults.processing_stats.average_time_per_document}s
                </div>
              </div>
              
              {/* Step Timings */}
              {currentResults.step_timings && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ marginBottom: '10px' }}>Step Breakdown:</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
                    {Object.entries(currentResults.step_timings).map(([step, duration]) => (
                      <div key={step} style={{ padding: '8px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
                        <strong style={{ textTransform: 'capitalize' }}>{step.replace('_', ' ')}:</strong> {duration}s
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Download Options */}
          {currentResults.archives && (
            <div className="result-card download-section">
              <h3>📦 Download Options</h3>
              
              {/* Master Archive - Primary Download */}
              {currentResults.archives.master_archive && (
                <div className="master-download">
                  <h4>🎁 Complete Package</h4>
                  <button
                    className="download-button master"
                    onClick={() => downloadArchive('master_archive')}
                  >
                    Download Master Archive (All Files Organized)
                  </button>
                  <p className="download-description">
                    Contains all cleaned PDFs, compressed versions, extractions, and organized folder structure
                  </p>
                </div>
              )}
              
              {/* Individual Downloads */}
              <div className="individual-downloads">
                <h4>📄 Individual Downloads</h4>
                <div className="download-buttons">
                  {Object.keys(currentResults.archives)
                    .filter(name => name !== 'master_archive')
                    .map(archiveName => {
                      let displayName = archiveName.replace(/_/g, ' ');
                      let description = '';
                      
                      if (archiveName === 'individual_uncompressed') {
                        displayName = 'Individual PDFs (Original Size)';
                        description = 'All split documents at original quality';
                      } else if (archiveName === 'individual_compressed') {
                        displayName = 'Individual PDFs (Compressed)';
                        description = `All split documents compressed to ${targetSizeMb || '2.0'}MB`;
                      } else if (archiveName === 'combined_uncompressed') {
                        displayName = 'Combined PDF (Original Size)';
                        description = 'All documents in one PDF';
                      } else if (archiveName === 'combined_compressed') {
                        displayName = 'Combined PDF (Compressed)';
                        description = 'All documents in one compressed PDF';
                      }
                      
                      return (
                        <div key={archiveName} className="download-item">
                          <button
                            className="download-button"
                            onClick={() => downloadArchive(archiveName)}
                          >
                            {displayName}
                          </button>
                          {description && <span className="download-item-description">{description}</span>}
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          )}
          
          {/* Extracted Data Preview */}
          {currentResults.extracted_documents && (
            <div className="result-card">
              <h3>📊 Extracted Data</h3>
              <details className="extraction-details">
                <summary>View extracted data ({currentResults.extracted_documents.length} documents)</summary>
                <pre className="json-preview">
                  {JSON.stringify(currentResults.extracted_documents, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PDFProcessorWorkflow;