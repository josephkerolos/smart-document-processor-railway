import React, { useState, useEffect, useRef } from 'react';
import './PDFProcessorWorkflow.css';

const PDFProcessorWorkflow_MultiFile = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSchema, setSelectedSchema] = useState('941-X');
  const [expectedValue, setExpectedValue] = useState('');
  const [targetSizeMb, setTargetSizeMb] = useState('9.0');
  const [fileResults, setFileResults] = useState({});
  const [activeFiles, setActiveFiles] = useState(new Set());
  const [elapsedTimes, setElapsedTimes] = useState({});
  const [combinedResults, setCombinedResults] = useState(null);
  const timerRefs = useRef({});
  const wsRefs = useRef({});

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      Object.values(timerRefs.current).forEach(clearInterval);
      Object.values(wsRefs.current).forEach(ws => ws?.close());
    };
  }, []);

  const updateElapsedTime = (fileId) => {
    setElapsedTimes(prev => ({
      ...prev,
      [fileId]: (prev[fileId] || 0) + 1
    }));
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setError(null);
    setFileResults({});
    setCombinedResults(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    setSelectedFiles(files);
    setError(null);
    setFileResults({});
    setCombinedResults(null);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const processFile = async (file, fileIndex, batchId = null) => {
    const fileId = `${file.name}_${fileIndex}`;
    
    // Start elapsed time tracking
    timerRefs.current[fileId] = setInterval(() => updateElapsedTime(fileId), 1000);
    
    setActiveFiles(prev => new Set([...prev, fileId]));
    setFileResults(prev => ({
      ...prev,
      [fileId]: {
        filename: file.name,
        status: 'processing',
        workflowSteps: [],
        sessionId: null,
        error: null,
        batchId: batchId
      }
    }));

    const formData = new FormData();
    formData.append('file', file);
    formData.append('selected_schema', selectedSchema);
    if (batchId) {
      formData.append('batch_id', batchId);
    }
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
      
      if (response.ok && result.session_id) {
        // Set session ID for this file
        setFileResults(prev => ({
          ...prev,
          [fileId]: {
            ...prev[fileId],
            sessionId: result.session_id
          }
        }));

        // Connect WebSocket for progress updates with retry logic
        const connectWebSocket = (retryCount = 0) => {
          const maxRetries = 3;
          const retryDelay = Math.min(1000 * Math.pow(2, retryCount), 5000); // Exponential backoff: 1s, 2s, 4s, max 5s
          
          // Add a small initial delay to ensure server is ready
          setTimeout(() => {
            const ws = new WebSocket(`ws://localhost:4830/ws/enhanced/${result.session_id}`);
            wsRefs.current[fileId] = ws;
            
            // Track if we should keep the WebSocket open
            let keepWsOpen = true;
            let connectionTimeout;

            // Set a timeout for connection
            connectionTimeout = setTimeout(() => {
              if (ws.readyState !== WebSocket.OPEN) {
                console.warn(`WebSocket connection timeout for ${file.name}, attempt ${retryCount + 1}/${maxRetries}`);
                ws.close();
                if (retryCount < maxRetries - 1) {
                  connectWebSocket(retryCount + 1);
                } else {
                  console.error(`Failed to establish WebSocket connection for ${file.name} after ${maxRetries} attempts`);
                  // Continue processing without real-time updates
                  setFileResults(prev => ({
                    ...prev,
                    [fileId]: {
                      ...prev[fileId],
                      error: 'WebSocket connection failed. Processing continues without real-time updates.'
                    }
                  }));
                }
              }
            }, 5000); // 5 second timeout for connection

            ws.onopen = () => {
              clearTimeout(connectionTimeout);
              console.log(`WebSocket connected for ${file.name}`);
            };

            ws.onmessage = (event) => {
              const data = JSON.parse(event.data);
              
              if (data.type === 'connection_established') {
                console.log(`WebSocket connection confirmed for ${file.name}`);
              } else if (data.type === 'ping') {
                // Respond to ping to keep connection alive
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send('pong');
                }
              } else if (data.type === 'workflow_plan') {
                setFileResults(prev => ({
                  ...prev,
                  [fileId]: {
                    ...prev[fileId],
                    workflowSteps: data.data.map(step => ({
                      id: step.id,
                      name: step.name,
                      status: step.status,
                      message: step.message || ''
                    }))
                  }
                }));
              } else if (data.type === 'step_update') {
                setFileResults(prev => ({
                  ...prev,
                  [fileId]: {
                    ...prev[fileId],
                    workflowSteps: prev[fileId]?.workflowSteps?.map(step =>
                      step.id === data.data.step_id
                        ? { ...step, status: data.data.status, message: data.data.message || '' }
                        : step
                    ) || []
                  }
                }));
                
                // If the complete step is done, we can close the WebSocket
                if (data.data.step_id === 'complete' && data.data.status === 'completed') {
                  keepWsOpen = false;
                  setTimeout(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                      ws.close();
                    }
                  }, 1000);
                }
              }
            };

            ws.onerror = (error) => {
              clearTimeout(connectionTimeout);
              if (keepWsOpen && retryCount < maxRetries - 1) {
                console.warn(`WebSocket error for ${file.name}, retrying...`);
                ws.close();
                connectWebSocket(retryCount + 1);
              } else if (keepWsOpen) {
                console.error(`WebSocket error for ${file.name}:`, error);
              }
            };

            ws.onclose = () => {
              clearTimeout(connectionTimeout);
              if (keepWsOpen && retryCount < maxRetries - 1) {
                console.log(`WebSocket closed for ${file.name}, retrying...`);
                connectWebSocket(retryCount + 1);
              } else {
                console.log(`WebSocket closed for ${file.name}`);
              }
            };
          }, retryCount === 0 ? 100 : retryDelay); // First attempt after 100ms, then exponential backoff
        };

        // Start WebSocket connection
        connectWebSocket();

        // The actual processing result comes from the HTTP response
        if (result.success) {
          clearInterval(timerRefs.current[fileId]);
          setFileResults(prev => ({
            ...prev,
            [fileId]: {
              ...prev[fileId],
              status: 'completed',
              result: result
            }
          }));
        } else {
          throw new Error(result.error || 'Processing failed');
        }
      } else {
        throw new Error(result.error || 'Processing failed');
      }
    } catch (err) {
      clearInterval(timerRefs.current[fileId]);
      setFileResults(prev => ({
        ...prev,
        [fileId]: {
          ...prev[fileId],
          status: 'failed',
          error: err.message
        }
      }));
    } finally {
      setActiveFiles(prev => {
        const newSet = new Set(prev);
        newSet.delete(fileId);
        return newSet;
      });
      
      // Close WebSocket if still open
      if (wsRefs.current[fileId]) {
        const ws = wsRefs.current[fileId];
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
        delete wsRefs.current[fileId];
      }
    }
  };

  const startProcessing = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one file');
      return;
    }

    setProcessing(true);
    setError(null);
    setFileResults({});
    setCombinedResults(null);
    setElapsedTimes({});

    try {
      let batchId = null;
      
      // Initialize batch only if multiple files (2 or more)
      if (selectedFiles.length > 1) {
        const batchResponse = await fetch('http://localhost:4830/api/init-batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_count: selectedFiles.length })
        });
        
        if (batchResponse.ok) {
          const batchData = await batchResponse.json();
          batchId = batchData.batch_id;
          console.log(`Initialized batch ${batchId} for ${selectedFiles.length} files`);
        }
      }
      
      // Process all files in parallel
      const processingPromises = selectedFiles.map((file, index) => 
        processFile(file, index, batchId)
      );

      await Promise.all(processingPromises);

      // After all files are processed, finalize batch only if we have a batch (multiple files)
      if (batchId) {
        // Wait a bit to ensure all backend processing is complete
        setTimeout(async () => {
          console.log('All files processed, finalizing batch...');
          await finalizeBatch(batchId);
          setProcessing(false);
        }, 1000);
      } else {
        // Single file processing - already complete with individual Google Drive upload
        setProcessing(false);
      }
    } catch (err) {
      setError(`Failed to process files: ${err.message}`);
      setProcessing(false);
    }
  };

  const finalizeBatch = async (batchId) => {
    try {
      const response = await fetch('http://localhost:4830/api/finalize-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch_id: batchId })
      });

      if (response.ok) {
        const result = await response.json();
        setCombinedResults(result);
        console.log('Batch finalized successfully:', result);
      } else {
        const error = await response.json();
        console.error('Failed to finalize batch:', error);
        setError(error.error || 'Failed to finalize batch');
      }
    } catch (err) {
      console.error('Failed to finalize batch:', err);
      setError('Failed to finalize batch: ' + err.message);
    }
  };


  const downloadArchive = (sessionId, archiveName) => {
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

  const allCompleted = Object.values(fileResults).every(r => r.status === 'completed' || r.status === 'failed');
  const anySuccessful = Object.values(fileResults).some(r => r.status === 'completed');

  return (
    <div className="pdf-processor-workflow">
      <div className="workflow-header">
        <h1>📄 Enhanced Document Workflow</h1>
        <p className="workflow-description">
          Clean, split, extract, organize, and compress IRS documents - supports both single and multiple files
        </p>
      </div>

      {/* Configuration Section */}
      <div className="config-section">
        <div className="config-grid">
          <div className="config-item">
            <label>Document Type</label>
            <select 
              value={selectedSchema} 
              onChange={(e) => setSelectedSchema(e.target.value)}
              disabled={processing}
            >
              <option value="941-X">IRS Form 941-X</option>
              <option value="941">IRS Form 941</option>
              <option value="1040">IRS Form 1040</option>
              <option value="2848">IRS Form 2848</option>
              <option value="8821">IRS Form 8821</option>
              <option value="payroll">Payroll Data</option>
              <option value="generic">Generic Document</option>
            </select>
          </div>

          <div className="config-item">
            <label>Expected Value (Optional)</label>
            <input
              type="text"
              value={expectedValue}
              onChange={(e) => setExpectedValue(e.target.value)}
              placeholder="e.g., 5107.23"
              disabled={processing}
            />
          </div>

          <div className="config-item">
            <label>Target Size (MB)</label>
            <input
              type="number"
              value={targetSizeMb}
              onChange={(e) => setTargetSizeMb(e.target.value)}
              placeholder="9.0"
              step="0.5"
              min="0.5"
              max="20"
              disabled={processing}
            />
          </div>
        </div>
      </div>

      {/* File Upload Section */}
      <div className="upload-section">
        <div 
          className={`drop-zone ${processing ? 'disabled' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => !processing && document.getElementById('file-input').click()}
        >
          <input
            id="file-input"
            type="file"
            multiple
            accept=".pdf"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            disabled={processing}
          />
          
          <div className="drop-zone-content">
            {selectedFiles.length === 0 ? (
              <>
                <div className="upload-icon">📤</div>
                <h3>Drop PDF files here or click to browse</h3>
                <p>Support for multiple IRS forms • Process in parallel</p>
              </>
            ) : (
              <>
                <div className="files-selected-icon">📄</div>
                <h3>{selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} selected</h3>
                <div className="selected-files-list">
                  {selectedFiles.map((file, idx) => (
                    <div key={idx} className="selected-file-item">
                      {file.name} ({(file.size / 1024 / 1024).toFixed(1)}MB)
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        <button
          className={`process-button ${processing ? 'processing' : ''}`}
          onClick={startProcessing}
          disabled={processing || selectedFiles.length === 0}
        >
          {processing ? '⏳ Processing...' : '🚀 Start Processing'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}

      {/* Individual File Progress */}
      {Object.entries(fileResults).map(([fileId, fileResult]) => (
        <div key={fileId} className="workflow-progress file-progress">
          <h3 className="file-progress-title">
            📄 {fileResult.filename}
            {fileResult.status === 'processing' && (
              <span className="elapsed-time">
                {formatTime(elapsedTimes[fileId] || 0)}
              </span>
            )}
          </h3>

          {fileResult.error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {fileResult.error}
            </div>
          )}

          {fileResult.workflowSteps.length > 0 && (
            <div className="workflow-steps">
              {fileResult.workflowSteps.map((step) => (
                <div key={step.id} className={`workflow-step ${step.status}`}>
                  <span className="step-icon">{getStepIcon(step.status)}</span>
                  <div className="step-content">
                    <span className="step-name">{step.name}</span>
                    {step.message && <span className="step-message">{step.message}</span>}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Individual File Results */}
          {fileResult.status === 'completed' && fileResult.result && (
            <div className="file-results">
              {fileResult.result.archives && (
                <div className="result-card download-section">
                  <h4>📦 Downloads for {fileResult.filename}</h4>
                  <div className="download-buttons">
                    {Object.keys(fileResult.result.archives).map(archiveName => (
                      <button
                        key={archiveName}
                        className="download-button"
                        onClick={() => downloadArchive(fileResult.sessionId, fileResult.result.archives[archiveName])}
                      >
                        {archiveName.replace(/_/g, ' ')}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Google Drive Upload for Individual File */}
              {fileResult.result.google_drive_upload?.success && (
                <div className="result-card" style={{ marginTop: '15px' }}>
                  <h4>☁️ Google Drive Upload</h4>
                  <p>Successfully uploaded to Google Drive</p>
                  <a 
                    href={fileResult.result.google_drive_upload.folder_link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="google-drive-link"
                  >
                    View in Google Drive →
                  </a>
                </div>
              )}
            </div>
          )}
        </div>
      ))}

      {/* Combined Results (when multiple files processed with batch) */}
      {allCompleted && anySuccessful && selectedFiles.length > 1 && combinedResults && (
        <div className="combined-results">
          <h2>🎯 Combined Results</h2>
          
          {/* Summary Statistics */}
          <div className="result-card">
            <h3>📊 Processing Summary</h3>
            <div className="stats-grid">
              <div>
                <strong>Total Files:</strong> {combinedResults.total_files}
              </div>
              <div>
                <strong>Total Documents:</strong> {combinedResults.total_documents}
              </div>
              <div>
                <strong>Total Pages Removed:</strong> {combinedResults.total_pages_removed}
              </div>
              <div>
                <strong>Combined Processing Time:</strong> {combinedResults.total_processing_time}s
              </div>
            </div>
            {combinedResults.company_names && combinedResults.company_names.length > 0 && (
              <div style={{ marginTop: '10px' }}>
                <strong>Company:</strong> {combinedResults.company_names.join(', ')}
              </div>
            )}
            {combinedResults.form_types && combinedResults.form_types.length > 0 && (
              <div>
                <strong>Form Types:</strong> {combinedResults.form_types.join(', ')}
              </div>
            )}
            {combinedResults.batch_folder && (
              <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#666' }}>
                <strong>Batch Location:</strong> {combinedResults.batch_folder}
              </div>
            )}
          </div>

          {/* Combined Downloads */}
          {combinedResults.combined_archives && (
            <div className="result-card download-section">
              <h3>📦 Combined Downloads</h3>
              <div className="master-download">
                <button
                  className="download-button master"
                  onClick={() => downloadArchive(combinedResults.session_id, combinedResults.combined_archives.master_archive)}
                >
                  Download Combined Master Archive
                </button>
                <p className="download-description">
                  Contains all processed files from all uploads, organized and combined
                </p>
              </div>
              
              {combinedResults.combined_archives.all_extractions && (
                <button
                  className="download-button"
                  onClick={() => downloadArchive(combinedResults.session_id, combinedResults.combined_archives.all_extractions)}
                >
                  Download All Extractions (JSON)
                </button>
              )}
              
              {combinedResults.all_extractions_path && (
                <div style={{ marginTop: '15px' }}>
                  <h4>📋 All Extractions JSON</h4>
                  <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>
                    All quarterly extractions combined as if they were in a single PDF
                  </p>
                  <button
                    className="download-button"
                    onClick={() => {
                      const url = `http://localhost:4830/api/download-file?path=${encodeURIComponent(combinedResults.all_extractions_path)}`;
                      window.open(url, '_blank');
                    }}
                  >
                    Download all_extractions.json
                  </button>
                </div>
              )}
              
              {combinedResults.master_archive && (
                <div style={{ marginTop: '15px' }}>
                  <h4>📦 Master Archive</h4>
                  <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>
                    Contains all processed files from all quarters
                  </p>
                  <button
                    className="download-button"
                    onClick={() => {
                      const url = `http://localhost:4830/api/download-file?path=${encodeURIComponent(combinedResults.master_archive)}`;
                      window.open(url, '_blank');
                    }}
                  >
                    Download Master Archive
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Google Drive Upload */}
          {combinedResults.google_drive_upload?.success && (
            <div className="result-card">
              <h3>☁️ Google Drive Upload</h3>
              <p>Successfully uploaded to Google Drive</p>
              <a 
                href={combinedResults.google_drive_upload.folder_link} 
                target="_blank" 
                rel="noopener noreferrer"
                className="google-drive-link"
              >
                View in Google Drive →
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PDFProcessorWorkflow_MultiFile;