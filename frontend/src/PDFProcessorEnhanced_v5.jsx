import React, { useState, useRef, useEffect } from "react";
import "./PDFProcessorEnhanced_v3.css";
import SchemaSelector from "./SchemaSelector";

const PDFProcessorEnhanced = () => {
  const [files, setFiles] = useState([]);
  const [selectedSchema, setSelectedSchema] = useState("generic");
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);
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
  
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const countdownInterval = useRef(null);
  const planContainerRef = useRef(null);
  const currentStepRef = useRef(null);
  const confirmationHandled = useRef(false);

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
    
    const websocket = new WebSocket(`ws://localhost:4830/ws/${sessionId}`);
    
    websocket.onopen = () => {
      console.log("WebSocket connected");
      setWs(websocket);
    };

    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
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
            
            // Start countdown for high confidence detections
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
          
        default:
          console.log("Unknown message type:", message.type);
      }
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setError("Connection error. Please refresh and try again.");
    };

    websocket.onclose = () => {
      console.log("WebSocket closed");
      if (processing) {
        setError("Connection lost. Processing may have been interrupted.");
        setProcessing(false);
      }
    };

    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
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
    
    setFiles(prevFiles => [...prevFiles, ...validFiles]);
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const processDocuments = async () => {
    if (files.length === 0) return;

    setProcessing(true);
    setProcessingPlan([]);
    setResults([]);
    setCurrentProgress(null);
    setShowFormConfirmation(null);
    setStatusMessages({});
    setElapsedTime(0);
    setError(null);
    confirmationHandled.current = false;
    stopCountdown();
    
    // Ensure WebSocket is connected
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("WebSocket not connected. Please refresh the page.");
      setProcessing(false);
      return;
    }

    const formData = new FormData();
    files.forEach(file => {
      formData.append("files", file);
    });
    formData.append("schema", selectedSchema);
    formData.append("session_id", sessionId);

    try {
      const response = await fetch("http://localhost:4830/process-documents-v3", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results || []);
      
      if (!data.results || data.results.length === 0) {
        setError("No results were extracted. Please check your document and try again.");
      }
    } catch (error) {
      console.error("Processing error:", error);
      setError(`Error processing documents: ${error.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const cancelProcessing = async () => {
    if (!sessionId) return;
    
    try {
      await fetch(`http://localhost:4830/cancel-processing/${sessionId}`, {
        method: "POST"
      });
      
      setProcessing(false);
      setProcessingPlan([]);
      setCurrentProgress(null);
      stopCountdown();
    } catch (error) {
      console.error("Error cancelling:", error);
    }
  };

  const handleFormConfirmation = async (useDetectedForm, isAuto = false) => {
    if (!showFormConfirmation || confirmationHandled.current) return;
    
    stopCountdown();
    confirmationHandled.current = true;
    
    const detectedForm = showFormConfirmation.detected_form;
    
    if (useDetectedForm) {
      setSelectedSchema(detectedForm);
    }
    
    // Send confirmation to backend
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

  return (
    <div className="pdf-processor-enhanced">
      <div className="processor-header">
        <h1>Intelligent Document Processing</h1>
        <p className="subtitle">AI-powered extraction with real-time insights</p>
      </div>

      <div className="processor-content">
        <div className="left-panel">
          {/* Schema Selector */}
          <div className="schema-section">
            <SchemaSelector 
              selectedSchema={selectedSchema} 
              onSchemaChange={setSelectedSchema} 
            />
          </div>

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

          {/* File List */}
          {files.length > 0 && (
            <div className="file-list">
              <h3>Selected Files ({files.length})</h3>
              {files.map((file, index) => (
                <div key={index} className="file-item">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                  <button 
                    className="remove-file"
                    onClick={() => removeFile(index)}
                    disabled={processing}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Enhanced Process/Cancel Button */}
          <div className="action-button-container">
            <button
              className={`action-button ${processing ? 'cancel' : 'process'}`}
              onClick={processing ? cancelProcessing : processDocuments}
              disabled={!processing && files.length === 0}
            >
              <span className="button-icon">
                {processing ? '🛑' : '📊'}
              </span>
              <span className="button-text">
                {processing ? 'Stop Processing' : 'Process Documents'}
              </span>
            </button>
          </div>
        </div>

        <div className="right-panel">
          {/* Time Elapsed Display */}
          {processing && (
            <div className="time-elapsed">
              <span className="time-label">Time Elapsed:</span>
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

          {/* Form Confirmation Dialog */}
          {showFormConfirmation && !confirmationHandled.current && (
            <div className="confirmation-dialog">
              <h3>Form Type Detected</h3>
              <p>
                We detected a <strong>{showFormConfirmation.detected_form}</strong> form 
                ({showFormConfirmation.confidence} confidence).
              </p>
              <p>{showFormConfirmation.message}</p>
              
              {confirmationCountdown !== null && (
                <div className="countdown-notice">
                  Auto-confirming in <strong>{confirmationCountdown}</strong> seconds...
                </div>
              )}
              
              <div className="confirmation-buttons">
                <button 
                  className="confirm-yes"
                  onClick={() => handleFormConfirmation(true)}
                >
                  Yes, use {showFormConfirmation.detected_form} template
                </button>
                <button 
                  className="confirm-no"
                  onClick={() => handleFormConfirmation(false)}
                >
                  No, keep generic template
                </button>
              </div>
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
                    
                    return (
                      <div 
                        key={index} 
                        ref={isCurrentStep ? currentStepRef : null}
                        className={`plan-step ${step.status} ${isCompleted ? 'minimized' : ''}`}
                      >
                        <div className="step-main">
                          <span className="step-icon">{getStepIcon(step.status)}</span>
                          <span className="step-name">{step.name}</span>
                          {step.message && step.status !== "in_progress" && (
                            <span className="step-message">{step.message}</span>
                          )}
                        </div>
                        
                        {/* Status line for current step */}
                        {isCurrentStep && statusMessages[step.id] && (
                          <div className="step-status-line">
                            → {statusMessages[step.id]}
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

          {/* Results */}
          {results.length > 0 && (
            <div className="results-section">
              <h3>Extraction Results</h3>
              {results.map((result, index) => (
                <div key={index} className="result-item">
                  <div className="result-header">
                    <h4>
                      Document {index + 1}
                      {result._metadata?.detected_form_type && 
                        ` - ${result._metadata.detected_form_type}`}
                    </h4>
                    <span className="result-meta">
                      Pages {result._metadata?.pages} • 
                      {result._metadata?.page_count} pages • 
                      Quality: {result._metadata?.quality_score?.toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="result-actions">
                    <button 
                      className="view-json"
                      onClick={() => {
                        const jsonStr = JSON.stringify(result, null, 2);
                        const blob = new Blob([jsonStr], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        window.open(url, '_blank');
                      }}
                    >
                      View JSON
                    </button>
                    <button 
                      className="download-json"
                      onClick={() => {
                        const jsonStr = JSON.stringify(result, null, 2);
                        const blob = new Blob([jsonStr], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `document_${index + 1}.json`;
                        a.click();
                      }}
                    >
                      Download
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PDFProcessorEnhanced;