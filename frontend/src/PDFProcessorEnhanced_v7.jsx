import React, { useState, useRef, useEffect } from "react";
import "./PDFProcessorEnhanced_v7.css";
import SchemaSelector from "./SchemaSelector";

const PDFProcessorEnhanced = () => {
  // File queue states
  const [fileQueue, setFileQueue] = useState([]);
  const [processingIndex, setProcessingIndex] = useState(-1);
  const [selectedSchema, setSelectedSchema] = useState("generic");
  const [processing, setProcessing] = useState(false);
  const [processingPlan, setProcessingPlan] = useState([]);
  const [currentProgress, setCurrentProgress] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [ws, setWs] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [showFormConfirmation, setShowFormConfirmation] = useState(null);
  const [confirmationCountdown, setConfirmationCountdown] = useState(null);
  const [statusMessages, setStatusMessages] = useState({});
  const [elapsedTime, setElapsedTime] = useState(0);
  const [error, setError] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [expectedValue, setExpectedValue] = useState("");
  const [matchResult, setMatchResult] = useState(null);
  const [expectedValueConfirmed, setExpectedValueConfirmed] = useState(false);
  const [currentFileStartTime, setCurrentFileStartTime] = useState(null);
  
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const countdownInterval = useRef(null);
  const planContainerRef = useRef(null);
  const currentStepRef = useRef(null);
  const confirmationHandled = useRef(false);
  const autoConfirmed = useRef(false);
  const processingInterrupted = useRef(false);

  // File status constants
  const FILE_STATUS = {
    PENDING: 'pending',
    PROCESSING: 'processing',
    COMPLETED: 'completed',
    FAILED: 'failed',
    CANCELLED: 'cancelled'
  };

  // Generate session ID
  useEffect(() => {
    setSessionId(generateSessionId());
  }, []);

  const generateSessionId = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  };

  // Format elapsed time
  const formatElapsedTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Auto-scroll to current step
  useEffect(() => {
    if (currentStepRef.current && planContainerRef.current) {
      const container = planContainerRef.current;
      const element = currentStepRef.current;
      const elementTop = element.offsetTop;
      const elementHeight = element.offsetHeight;
      const containerHeight = container.offsetHeight;
      const scrollTop = elementTop - (containerHeight / 2) + (elementHeight / 2);
      
      container.scrollTo({
        top: scrollTop,
        behavior: 'smooth'
      });
    }
  }, [processingPlan]);

  // WebSocket connection
  useEffect(() => {
    if (!sessionId) return;
    
    let websocket = null;
    let heartbeatInterval = null;
    let reconnectTimeout = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    
    const connectWebSocket = () => {
      const wsUrl = `ws://localhost:4830/ws/${sessionId}`;
      console.log('Attempting WebSocket connection to:', wsUrl);
      websocket = new WebSocket(wsUrl);
    
      websocket.onopen = () => {
        console.log("WebSocket connected");
        setWs(websocket);
        setWsConnected(true);
        setError(null);
        reconnectAttempts = 0;
        
        heartbeatInterval = setInterval(() => {
          if (websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };

      websocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
      
        switch (message.type) {
          case "connection_established":
            console.log("Server confirmed connection:", message);
            break;
            
          case "processing_plan":
            setProcessingPlan(message.data.steps);
            break;
            
          case "progress":
            setCurrentProgress(message.data);
            break;
            
          case "status_update":
            setStatusMessages(prev => ({
              ...prev,
              [message.data.step_id]: message.data.message
            }));
            break;
            
          case "time_update":
            setElapsedTime(message.data.elapsed_seconds);
            break;
            
          case "form_confirmation_needed":
            if (!confirmationHandled.current) {
              const confirmData = message.data;
              setShowFormConfirmation(confirmData);
              
              if (confirmData.confidence === "high") {
                setConfirmationCountdown(5);
                startCountdown();
              }
            }
            break;
            
          case "confirmation_received":
            confirmationHandled.current = true;
            setShowFormConfirmation(null);
            stopCountdown();
            break;
            
          case "cancelled":
            setProcessing(false);
            break;
            
          case "pong":
            break;
            
          case "value_match_result":
            setMatchResult(message.data);
            break;
            
          default:
            console.log("Unknown message type:", message.type);
        }
      };

      websocket.onerror = (error) => {
        console.error("WebSocket error:", error);
        setWsConnected(false);
        if (reconnectAttempts >= maxReconnectAttempts) {
          setError("Unable to connect. Please refresh and try again.");
        }
      };

      websocket.onclose = (event) => {
        console.log("WebSocket closed", event.code, event.reason);
        setWsConnected(false);
        
        if (event.code === 1000 || processing) {
          if (processing && event.code !== 1000) {
            setError("Connection lost. Processing may have been interrupted.");
            setProcessing(false);
            processingInterrupted.current = true;
          }
          return;
        }
        
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), 10000);
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
          reconnectTimeout = setTimeout(connectWebSocket, delay);
        }
      };
    };
    
    connectWebSocket();

    return () => {
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  }, [sessionId]);

  // Countdown timer for auto-confirmation
  const startCountdown = () => {
    if (countdownInterval.current) {
      clearInterval(countdownInterval.current);
    }
    
    countdownInterval.current = setInterval(() => {
      setConfirmationCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownInterval.current);
          handleFormConfirmation(true, true);
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const stopCountdown = () => {
    if (countdownInterval.current) {
      clearInterval(countdownInterval.current);
      countdownInterval.current = null;
    }
    setConfirmationCountdown(null);
  };

  // Cleanup countdown on unmount
  useEffect(() => {
    return () => {
      if (countdownInterval.current) {
        clearInterval(countdownInterval.current);
      }
    };
  }, []);

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragIn = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setDragActive(true);
    }
  };

  const handleDragOut = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFiles = (fileList) => {
    const validFiles = fileList.filter(file => 
      file.type === 'application/pdf' || file.type.startsWith('image/')
    );
    
    if (validFiles.length !== fileList.length) {
      alert("Some files were skipped. Only PDF and image files are supported.");
    }
    
    // Add files to queue with metadata
    const newFiles = validFiles.map((file, index) => ({
      id: Date.now() + index,
      file: file,
      status: FILE_STATUS.PENDING,
      result: null,
      error: null,
      startTime: null,
      endTime: null,
      elapsedTime: null
    }));
    
    setFileQueue(prev => [...prev, ...newFiles]);
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const removeFile = (fileId) => {
    setFileQueue(prev => prev.filter(f => f.id !== fileId));
  };

  const clearCompleted = () => {
    setFileQueue(prev => prev.filter(f => f.status !== FILE_STATUS.COMPLETED));
  };

  const clearAll = () => {
    if (processing) {
      if (window.confirm("This will stop current processing. Continue?")) {
        cancelProcessing();
        setFileQueue([]);
      }
    } else {
      setFileQueue([]);
    }
  };

  // Process next file in queue
  const processNextFile = async () => {
    const nextIndex = fileQueue.findIndex(f => f.status === FILE_STATUS.PENDING);
    
    if (nextIndex === -1 || processingInterrupted.current) {
      // No more files to process
      setProcessing(false);
      setProcessingIndex(-1);
      processingInterrupted.current = false;
      return;
    }

    setProcessingIndex(nextIndex);
    const fileItem = fileQueue[nextIndex];
    
    // Update file status to processing
    setFileQueue(prev => prev.map((f, idx) => 
      idx === nextIndex ? { ...f, status: FILE_STATUS.PROCESSING, startTime: Date.now() } : f
    ));
    
    // Reset states for new file
    setProcessingPlan([]);
    setMatchResult(null);
    setCurrentProgress(null);
    setShowFormConfirmation(null);
    setStatusMessages({});
    setElapsedTime(0);
    setError(null);
    confirmationHandled.current = false;
    autoConfirmed.current = false;
    stopCountdown();
    setCurrentFileStartTime(Date.now());

    const formData = new FormData();
    formData.append("files", fileItem.file);
    formData.append("schema", selectedSchema);
    formData.append("session_id", sessionId);
    if (expectedValue) {
      formData.append("expected_value", expectedValue);
    }

    try {
      const response = await fetch("http://localhost:4830/process-documents-v3", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      // Update file with result
      setFileQueue(prev => prev.map((f, idx) => 
        idx === nextIndex ? { 
          ...f, 
          status: FILE_STATUS.COMPLETED, 
          result: data.results?.[0] || null,
          endTime: Date.now(),
          elapsedTime: Date.now() - f.startTime
        } : f
      ));
      
      // Process next file
      setTimeout(() => processNextFile(), 500);
      
    } catch (error) {
      console.error("Processing error:", error);
      
      // Update file with error
      setFileQueue(prev => prev.map((f, idx) => 
        idx === nextIndex ? { 
          ...f, 
          status: FILE_STATUS.FAILED, 
          error: error.message,
          endTime: Date.now(),
          elapsedTime: Date.now() - f.startTime
        } : f
      ));
      
      // Continue with next file
      setTimeout(() => processNextFile(), 500);
    }
  };

  const startProcessing = async () => {
    if (fileQueue.filter(f => f.status === FILE_STATUS.PENDING).length === 0) return;

    setProcessing(true);
    processingInterrupted.current = false;
    
    // Ensure WebSocket is connected
    if (!wsConnected || !ws || ws.readyState !== WebSocket.OPEN) {
      setError("WebSocket not connected. Please wait a moment and try again.");
      setProcessing(false);
      return;
    }

    processNextFile();
  };

  const cancelProcessing = async () => {
    processingInterrupted.current = true;
    
    if (!sessionId) return;
    
    try {
      await fetch(`http://localhost:4830/cancel-processing/${sessionId}`, {
        method: "POST"
      });
      
      // Mark current processing file as cancelled
      if (processingIndex >= 0) {
        setFileQueue(prev => prev.map((f, idx) => 
          idx === processingIndex ? { 
            ...f, 
            status: FILE_STATUS.CANCELLED,
            endTime: Date.now(),
            elapsedTime: Date.now() - f.startTime
          } : f
        ));
      }
      
      setProcessing(false);
      setProcessingPlan([]);
      setCurrentProgress(null);
      setProcessingIndex(-1);
      stopCountdown();
    } catch (error) {
      console.error("Error cancelling:", error);
    }
  };

  const handleFormConfirmation = async (useDetectedForm, isAuto = false) => {
    if (!showFormConfirmation || confirmationHandled.current) return;
    
    stopCountdown();
    confirmationHandled.current = true;
    if (isAuto) {
      autoConfirmed.current = true;
    }
    
    const detectedForm = showFormConfirmation.detected_form;
    
    if (useDetectedForm) {
      setSelectedSchema(detectedForm);
    }
    
    try {
      await fetch("http://localhost:4830/confirm-form-selection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          use_detected_form: useDetectedForm,
          detected_form: detectedForm,
          auto_confirmed: isAuto
        })
      });
    } catch (error) {
      console.error("Error confirming form selection:", error);
    }
    
    setShowFormConfirmation(null);
  };

  const getStepIcon = (status) => {
    switch (status) {
      case "completed": return "✅";
      case "in_progress": return "⏳";
      case "failed": return "❌";
      case "cancelled": return "🚫";
      case "waiting": return "⏸️";
      default: return "⚪";
    }
  };

  const getFileStatusIcon = (status) => {
    switch (status) {
      case FILE_STATUS.COMPLETED: return "✅";
      case FILE_STATUS.PROCESSING: return "⏳";
      case FILE_STATUS.FAILED: return "❌";
      case FILE_STATUS.CANCELLED: return "🚫";
      default: return "⏸️";
    }
  };

  const getQueueStats = () => {
    const total = fileQueue.length;
    const pending = fileQueue.filter(f => f.status === FILE_STATUS.PENDING).length;
    const completed = fileQueue.filter(f => f.status === FILE_STATUS.COMPLETED).length;
    const failed = fileQueue.filter(f => f.status === FILE_STATUS.FAILED).length;
    
    return { total, pending, completed, failed };
  };

  return (
    <div className="pdf-processor-enhanced v7">
      <div className="processor-header">
        <h1>Intelligent Document Processing</h1>
        <p className="subtitle">AI-powered extraction with file queue management</p>
        <div className="connection-status">
          <span className={`status-indicator ${wsConnected ? 'connected' : 'disconnected'}`}>
            {wsConnected ? '● Connected' : '● Disconnected'}
          </span>
        </div>
      </div>

      <div className="processor-content">
        <div className="left-panel">
          {/* Schema Selector */}
          <div className="schema-section">
            <SchemaSelector 
              selectedSchema={selectedSchema} 
              onSchemaChange={setSelectedSchema}
              disabled={processing || autoConfirmed.current}
            />
          </div>

          {/* Expected Value Input - For 941-X ERC matching */}
          {(selectedSchema === "941-X" || (showFormConfirmation && showFormConfirmation.detected_form === "941-X")) && (
            <div className="expected-value-section">
              <label htmlFor="expected-value">
                Expected ERC Amount (Optional):
              </label>
              <div className="expected-value-controls">
                <input
                  id="expected-value"
                  type="text"
                  placeholder="e.g., 5107.23"
                  value={expectedValue}
                  onChange={(e) => {
                    setExpectedValue(e.target.value);
                    setExpectedValueConfirmed(false);
                  }}
                  className="expected-value-input"
                />
                <button
                  className="expected-value-submit"
                  onClick={() => {
                    if (expectedValue.trim()) {
                      setExpectedValueConfirmed(true);
                    }
                  }}
                  disabled={!expectedValue.trim()}
                >
                  Set Value
                </button>
              </div>
              {expectedValueConfirmed && expectedValue && (
                <div className="expected-value-confirmed">
                  ✅ Expected value set: ${expectedValue}
                </div>
              )}
              <small>Enter the Employee Retention Credit amount you expect to find</small>
            </div>
          )}

          {/* File Upload Zone */}
          <div 
            ref={dropZoneRef}
            className={`drop-zone ${dragActive ? 'drag-active' : ''} ${processing ? 'disabled' : ''}`}
            onDragEnter={handleDragIn}
            onDragLeave={handleDragOut}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => !processing && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
              disabled={processing}
            />
            
            <div className="drop-zone-content">
              <div className="upload-icon">📄</div>
              <h3>Drop files here or click to browse</h3>
              <p>Support for PDF and image files. Multiple files allowed.</p>
            </div>
          </div>

          {/* File Queue */}
          {fileQueue.length > 0 && (
            <div className="file-queue">
              <div className="queue-header">
                <h3>File Queue ({fileQueue.length})</h3>
                <div className="queue-actions">
                  <button 
                    className="clear-completed"
                    onClick={clearCompleted}
                    disabled={fileQueue.filter(f => f.status === FILE_STATUS.COMPLETED).length === 0}
                  >
                    Clear Completed
                  </button>
                  <button 
                    className="clear-all"
                    onClick={clearAll}
                  >
                    Clear All
                  </button>
                </div>
              </div>
              
              <div className="queue-stats">
                {(() => {
                  const stats = getQueueStats();
                  return (
                    <>
                      <span className="stat pending">⏸️ {stats.pending} pending</span>
                      <span className="stat completed">✅ {stats.completed} completed</span>
                      {stats.failed > 0 && <span className="stat failed">❌ {stats.failed} failed</span>}
                    </>
                  );
                })()}
              </div>
              
              <div className="file-list">
                {fileQueue.map((fileItem) => (
                  <div key={fileItem.id} className={`file-item ${fileItem.status}`}>
                    <span className="file-status-icon">{getFileStatusIcon(fileItem.status)}</span>
                    <div className="file-info">
                      <span className="file-name">{fileItem.file.name}</span>
                      <span className="file-meta">
                        {(fileItem.file.size / 1024 / 1024).toFixed(2)} MB
                        {fileItem.elapsedTime && ` • ${formatElapsedTime(Math.floor(fileItem.elapsedTime / 1000))}`}
                      </span>
                    </div>
                    <div className="file-actions">
                      {fileItem.status === FILE_STATUS.COMPLETED && fileItem.result && (
                        <button 
                          className="view-result"
                          onClick={() => {
                            const jsonStr = JSON.stringify(fileItem.result, null, 2);
                            const blob = new Blob([jsonStr], { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            window.open(url, '_blank');
                          }}
                        >
                          View
                        </button>
                      )}
                      {fileItem.status === FILE_STATUS.PENDING && (
                        <button 
                          className="remove-file"
                          onClick={() => removeFile(fileItem.id)}
                          disabled={processing}
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Process/Cancel Button */}
          <div className="action-button-container">
            <button
              className={`action-button enhanced ${processing ? 'cancel' : 'process'}`}
              onClick={processing ? cancelProcessing : startProcessing}
              disabled={(!processing && fileQueue.filter(f => f.status === FILE_STATUS.PENDING).length === 0) || !wsConnected}
            >
              <span className="button-icon">
                {processing ? '🛑' : '🚀'}
              </span>
              <span className="button-text">
                {processing ? 'Stop Processing' : !wsConnected ? 'Connecting...' : 'Start Processing'}
              </span>
              {processing && <span className="button-pulse"></span>}
            </button>
          </div>
        </div>

        <div className="right-panel">
          {/* Currently Processing File */}
          {processing && processingIndex >= 0 && fileQueue[processingIndex] && (
            <div className="current-file-banner">
              <h3>Processing File {processingIndex + 1} of {fileQueue.length}</h3>
              <p>{fileQueue[processingIndex].file.name}</p>
              {currentFileStartTime && (
                <span className="file-timer">
                  ⏱️ {formatElapsedTime(Math.floor((Date.now() - currentFileStartTime) / 1000))}
                </span>
              )}
            </div>
          )}

          {/* Status Bar */}
          <div className="status-bar">
            {/* Time Elapsed Display */}
            {processing && (
              <div className="time-elapsed">
                <span className="time-icon">⏱️</span>
                <span className="time-value">{formatElapsedTime(elapsedTime)}</span>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="error-banner">
                <span className="error-icon">⚠️</span>
                <span className="error-message">{error}</span>
                <button 
                  className="error-close"
                  onClick={() => setError(null)}
                >
                  ✕
                </button>
              </div>
            )}
          </div>

          {/* Form Confirmation Dialog */}
          {showFormConfirmation && !confirmationHandled.current && (
            <div className="confirmation-dialog floating">
              <h3>Form Type Detected</h3>
              <p>
                Detected <strong>{showFormConfirmation.detected_form}</strong> 
                {showFormConfirmation.confidence === 'high' ? ' with high confidence' : ''}.
              </p>
              
              {confirmationCountdown !== null && (
                <div className="countdown-notice">
                  Auto-confirming in <strong>{confirmationCountdown}</strong>s
                </div>
              )}
              
              <div className="confirmation-buttons">
                <button 
                  className="confirm-yes"
                  onClick={() => handleFormConfirmation(true)}
                >
                  Use {showFormConfirmation.detected_form}
                </button>
                <button 
                  className="confirm-no"
                  onClick={() => handleFormConfirmation(false)}
                >
                  Keep generic
                </button>
              </div>
            </div>
          )}
          
          {/* Auto-confirmed notification */}
          {autoConfirmed.current && !processing && !showFormConfirmation && (
            <div className="auto-confirmed-notice">
              <span className="notice-icon">✅</span>
              <span className="notice-text">
                Template auto-selected: <strong>{selectedSchema}</strong>
              </span>
            </div>
          )}

          {/* Processing Plan */}
          {(processing || processingPlan.length > 0) && (
            <div className="processing-plan">
              <h3>Processing Progress</h3>
              <div className="plan-container" ref={planContainerRef}>
                {processingPlan.length > 0 ? (
                  processingPlan.map((step, index) => {
                    const isCurrentStep = step.status === "in_progress";
                    const isCompleted = step.status === "completed";
                    const isWaiting = step.status === "waiting";
                    
                    let displayMessage = step.message;
                    if (step.id === "validate" && isWaiting && autoConfirmed.current) {
                      displayMessage = "Auto-confirmed template selection";
                    }
                    
                    return (
                      <div 
                        key={index} 
                        ref={isCurrentStep ? currentStepRef : null}
                        className={`plan-step ${step.status} ${isCompleted ? 'minimized' : ''}`}
                      >
                        <div className="step-main">
                          <span className="step-icon">{getStepIcon(step.status)}</span>
                          <span className="step-name">{step.name}</span>
                          {displayMessage && (
                            <span className="step-message">- {displayMessage}</span>
                          )}
                        </div>
                        
                        {isCurrentStep && statusMessages[step.id] && (
                          <div className="step-status-line">
                            <span className="status-prefix">→</span>
                            <span className="status-text">{statusMessages[step.id]}</span>
                          </div>
                        )}
                      </div>
                    );
                  })
                ) : processing ? (
                  <div className="plan-loading">
                    <span className="step-icon">⏳</span>
                    <span className="step-name">Initializing processing engine...</span>
                  </div>
                ) : null}
              </div>
              
              {/* Progress Bar */}
              {currentProgress && (
                <div className="progress-section">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${currentProgress.percentage || 0}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Match Result Display */}
          {matchResult && (
            <div className={`match-result ${matchResult.is_match ? 'match' : 'no-match'}`}>
              <h3>🎯 ERC Value Matching Result</h3>
              <div className="match-details">
                <div className="match-summary">
                  {matchResult.is_match ? (
                    <div className="match-success">
                      <span className="match-icon">✅</span>
                      <span className="match-text">MATCH FOUND!</span>
                    </div>
                  ) : (
                    <div className="match-failure">
                      <span className="match-icon">❌</span>
                      <span className="match-text">NO MATCH</span>
                    </div>
                  )}
                </div>
                <div className="match-values">
                  <div className="value-item">
                    <span className="value-label">Expected:</span>
                    <span className="value-amount">${matchResult.expected_value}</span>
                  </div>
                  <div className="value-item">
                    <span className="value-label">Found:</span>
                    <span className="value-amount">${matchResult.found_value || 'None'}</span>
                  </div>
                </div>
                {matchResult.field_path && (
                  <p className="field-location"><strong>Found in:</strong> {matchResult.field_path}</p>
                )}
                {matchResult.confidence && (
                  <p className="confidence-level"><strong>Confidence:</strong> {matchResult.confidence}</p>
                )}
                {matchResult.explanation && (
                  <div className="match-explanation">
                    <strong>Details:</strong> {matchResult.explanation}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PDFProcessorEnhanced;