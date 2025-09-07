import React, { useState, useRef, useEffect } from "react";
import "./PDFProcessorEnhanced_v8.css";
import SchemaSelector from "./SchemaSelector";

const PDFProcessorEnhanced = () => {
  // File queue states
  const [fileQueue, setFileQueue] = useState([]);
  const [currentProcessingId, setCurrentProcessingId] = useState(null);
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
  const [expandedFileId, setExpandedFileId] = useState(null);
  const [queueSection, setQueueSection] = useState('all'); // all, pending, processing, completed, failed
  
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const countdownInterval = useRef(null);
  const planContainerRef = useRef(null);
  const currentStepRef = useRef(null);
  const confirmationHandled = useRef(false);
  const autoConfirmed = useRef(false);
  const processingInterrupted = useRef(false);
  const processedFiles = useRef(new Set());

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

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
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
    
    // Check queue limit
    const currentQueueSize = fileQueue.length;
    const maxQueueSize = 100;
    const availableSlots = maxQueueSize - currentQueueSize;
    
    if (currentQueueSize >= maxQueueSize) {
      alert(`Queue is full. Maximum ${maxQueueSize} files allowed. Please clear some files before adding more.`);
      return;
    }
    
    if (validFiles.length > availableSlots) {
      const confirmAdd = window.confirm(
        `Queue limit is ${maxQueueSize} files. You have ${currentQueueSize} files in queue.\n` +
        `Only ${availableSlots} of ${validFiles.length} selected files will be added.\n\n` +
        `Continue?`
      );
      if (!confirmAdd) return;
    }
    
    // Only take files that fit in the queue
    const filesToAdd = validFiles.slice(0, availableSlots);
    
    // Add files to queue with metadata
    const newFiles = filesToAdd.map((file, index) => ({
      id: `${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
      file: file,
      status: FILE_STATUS.PENDING,
      result: null,
      error: null,
      startTime: null,
      endTime: null,
      elapsedTime: null,
      detectedForm: null,
      matchResult: null
    }));
    
    setFileQueue(prev => [...prev, ...newFiles]);
    
    if (validFiles.length > availableSlots) {
      alert(`Added ${filesToAdd.length} files to queue. ${validFiles.length - filesToAdd.length} files were skipped due to queue limit.`);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const removeFile = (fileId) => {
    setFileQueue(prev => prev.filter(f => f.id !== fileId));
    processedFiles.current.delete(fileId);
  };

  const clearCompleted = () => {
    const completedIds = fileQueue.filter(f => f.status === FILE_STATUS.COMPLETED).map(f => f.id);
    completedIds.forEach(id => processedFiles.current.delete(id));
    setFileQueue(prev => prev.filter(f => f.status !== FILE_STATUS.COMPLETED));
  };

  const clearAll = () => {
    if (processing) {
      if (window.confirm("This will stop current processing. Continue?")) {
        cancelProcessing();
        setFileQueue([]);
        processedFiles.current.clear();
      }
    } else {
      setFileQueue([]);
      processedFiles.current.clear();
    }
  };

  // Process next file in queue
  const processNextFile = async () => {
    // Find next unprocessed file
    const nextFile = fileQueue.find(f => 
      f.status === FILE_STATUS.PENDING && 
      !processedFiles.current.has(f.id)
    );
    
    if (!nextFile || processingInterrupted.current) {
      // No more files to process
      setProcessing(false);
      setCurrentProcessingId(null);
      processingInterrupted.current = false;
      console.log("Queue processing complete");
      return;
    }

    // Mark this file as being processed
    processedFiles.current.add(nextFile.id);
    setCurrentProcessingId(nextFile.id);
    
    // Update file status to processing
    setFileQueue(prev => prev.map(f => 
      f.id === nextFile.id ? { ...f, status: FILE_STATUS.PROCESSING, startTime: Date.now() } : f
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
    formData.append("files", nextFile.file);
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
      setFileQueue(prev => prev.map(f => 
        f.id === nextFile.id ? { 
          ...f, 
          status: FILE_STATUS.COMPLETED, 
          result: data.results?.[0] || null,
          endTime: Date.now(),
          elapsedTime: Date.now() - f.startTime,
          detectedForm: data.results?.[0]?._metadata?.detected_form_type || selectedSchema,
          matchResult: matchResult
        } : f
      ));
      
      console.log(`File ${nextFile.file.name} completed successfully`);
      
      // Process next file after a short delay
      setTimeout(() => processNextFile(), 1000);
      
    } catch (error) {
      console.error("Processing error:", error);
      
      // Update file with error
      setFileQueue(prev => prev.map(f => 
        f.id === nextFile.id ? { 
          ...f, 
          status: FILE_STATUS.FAILED, 
          error: error.message,
          endTime: Date.now(),
          elapsedTime: Date.now() - f.startTime
        } : f
      ));
      
      // Continue with next file after a short delay
      setTimeout(() => processNextFile(), 1000);
    }
  };

  const startProcessing = async () => {
    const pendingFiles = fileQueue.filter(f => 
      f.status === FILE_STATUS.PENDING && 
      !processedFiles.current.has(f.id)
    );
    
    if (pendingFiles.length === 0) {
      setError("No pending files to process");
      return;
    }

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
      if (currentProcessingId) {
        setFileQueue(prev => prev.map(f => 
          f.id === currentProcessingId ? { 
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
      setCurrentProcessingId(null);
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
    const processing = fileQueue.filter(f => f.status === FILE_STATUS.PROCESSING).length;
    const completed = fileQueue.filter(f => f.status === FILE_STATUS.COMPLETED).length;
    const failed = fileQueue.filter(f => f.status === FILE_STATUS.FAILED).length;
    
    return { total, pending, processing, completed, failed };
  };

  const getFilteredFiles = () => {
    switch (queueSection) {
      case 'pending':
        return fileQueue.filter(f => f.status === FILE_STATUS.PENDING);
      case 'processing':
        return fileQueue.filter(f => f.status === FILE_STATUS.PROCESSING);
      case 'completed':
        return fileQueue.filter(f => f.status === FILE_STATUS.COMPLETED);
      case 'failed':
        return fileQueue.filter(f => f.status === FILE_STATUS.FAILED || f.status === FILE_STATUS.CANCELLED);
      default:
        return fileQueue;
    }
  };

  const handleFileClick = (fileId) => {
    setExpandedFileId(expandedFileId === fileId ? null : fileId);
  };

  const retryFile = (fileId) => {
    processedFiles.current.delete(fileId);
    setFileQueue(prev => prev.map(f => 
      f.id === fileId ? { 
        ...f, 
        status: FILE_STATUS.PENDING,
        result: null,
        error: null,
        startTime: null,
        endTime: null,
        elapsedTime: null,
        detectedForm: null,
        matchResult: null
      } : f
    ));
  };

  return (
    <div className="pdf-processor-enhanced v8">
      <div className="processor-header">
        <h1>Intelligent Document Processing</h1>
        <p className="subtitle">AI-powered extraction with advanced queue management</p>
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

          {/* Queue Stats */}
          <div className="queue-stats-panel">
            {(() => {
              const stats = getQueueStats();
              return (
                <>
                  <div className="stat-item">
                    <span className="stat-value">{stats.total}</span>
                    <span className="stat-label">Total Files</span>
                  </div>
                  <div className="stat-item pending">
                    <span className="stat-value">{stats.pending}</span>
                    <span className="stat-label">Pending</span>
                  </div>
                  <div className="stat-item processing">
                    <span className="stat-value">{stats.processing}</span>
                    <span className="stat-label">Processing</span>
                  </div>
                  <div className="stat-item completed">
                    <span className="stat-value">{stats.completed}</span>
                    <span className="stat-label">Completed</span>
                  </div>
                  {stats.failed > 0 && (
                    <div className="stat-item failed">
                      <span className="stat-value">{stats.failed}</span>
                      <span className="stat-label">Failed</span>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        </div>

        <div className="right-panel">
          {/* File Queue with Tabs */}
          {fileQueue.length > 0 && (
            <div className="file-queue-container">
              <div className="queue-header">
                <h3>
                  File Queue ({fileQueue.length}/100)
                  {fileQueue.length >= 100 && <span className="queue-full-badge">FULL</span>}
                </h3>
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
              
              {/* Queue Tabs */}
              <div className="queue-tabs">
                <button 
                  className={`queue-tab ${queueSection === 'all' ? 'active' : ''}`}
                  onClick={() => setQueueSection('all')}
                >
                  All ({fileQueue.length})
                </button>
                <button 
                  className={`queue-tab ${queueSection === 'pending' ? 'active' : ''}`}
                  onClick={() => setQueueSection('pending')}
                >
                  Pending ({fileQueue.filter(f => f.status === FILE_STATUS.PENDING).length})
                </button>
                <button 
                  className={`queue-tab ${queueSection === 'processing' ? 'active' : ''}`}
                  onClick={() => setQueueSection('processing')}
                >
                  Processing ({fileQueue.filter(f => f.status === FILE_STATUS.PROCESSING).length})
                </button>
                <button 
                  className={`queue-tab ${queueSection === 'completed' ? 'active' : ''}`}
                  onClick={() => setQueueSection('completed')}
                >
                  Completed ({fileQueue.filter(f => f.status === FILE_STATUS.COMPLETED).length})
                </button>
                {fileQueue.filter(f => f.status === FILE_STATUS.FAILED || f.status === FILE_STATUS.CANCELLED).length > 0 && (
                  <button 
                    className={`queue-tab ${queueSection === 'failed' ? 'active' : ''}`}
                    onClick={() => setQueueSection('failed')}
                  >
                    Failed ({fileQueue.filter(f => f.status === FILE_STATUS.FAILED || f.status === FILE_STATUS.CANCELLED).length})
                  </button>
                )}
              </div>
              
              {/* File List */}
              <div className="file-list">
                {getFilteredFiles().map((fileItem) => (
                  <div 
                    key={fileItem.id} 
                    className={`file-card ${fileItem.status} ${expandedFileId === fileItem.id ? 'expanded' : ''} ${fileItem.id === currentProcessingId ? 'current' : ''}`}
                  >
                    <div 
                      className="file-header"
                      onClick={() => handleFileClick(fileItem.id)}
                    >
                      <span className="file-status-icon">{getFileStatusIcon(fileItem.status)}</span>
                      <div className="file-info">
                        <span className="file-name">{fileItem.file.name}</span>
                        <div className="file-meta">
                          <span>{formatFileSize(fileItem.file.size)}</span>
                          {fileItem.elapsedTime && <span> • {formatElapsedTime(Math.floor(fileItem.elapsedTime / 1000))}</span>}
                          {fileItem.detectedForm && <span> • {fileItem.detectedForm}</span>}
                        </div>
                      </div>
                      <span className="expand-icon">{expandedFileId === fileItem.id ? '▼' : '▶'}</span>
                    </div>
                    
                    {/* Expanded Details */}
                    {expandedFileId === fileItem.id && (
                      <div className="file-details">
                        {fileItem.status === FILE_STATUS.PROCESSING && (
                          <div className="processing-indicator">
                            <div className="processing-spinner"></div>
                            <span>Processing... {currentFileStartTime && formatElapsedTime(Math.floor((Date.now() - currentFileStartTime) / 1000))}</span>
                          </div>
                        )}
                        
                        {fileItem.error && (
                          <div className="error-details">
                            <strong>Error:</strong> {fileItem.error}
                          </div>
                        )}
                        
                        {fileItem.matchResult && (
                          <div className="match-result-mini">
                            <strong>ERC Match:</strong> {fileItem.matchResult.is_match ? '✅ Found' : '❌ Not Found'}
                          </div>
                        )}
                        
                        <div className="file-actions">
                          {fileItem.status === FILE_STATUS.COMPLETED && fileItem.result && (
                            <>
                              <button 
                                className="action-btn view"
                                onClick={() => {
                                  const jsonStr = JSON.stringify(fileItem.result, null, 2);
                                  const blob = new Blob([jsonStr], { type: 'application/json' });
                                  const url = URL.createObjectURL(blob);
                                  window.open(url, '_blank');
                                }}
                              >
                                View Result
                              </button>
                              <button 
                                className="action-btn download"
                                onClick={() => {
                                  const jsonStr = JSON.stringify(fileItem.result, null, 2);
                                  const blob = new Blob([jsonStr], { type: 'application/json' });
                                  const url = URL.createObjectURL(blob);
                                  const a = document.createElement('a');
                                  a.href = url;
                                  
                                  // Generate filename based on extracted data
                                  let filename = '';
                                  const result = fileItem.result;
                                  
                                  // Try to get employer name
                                  const employerName = result?.employerInfo?.name || 
                                                      result?.employer_info?.name || 
                                                      result?.business_name ||
                                                      result?.company_name ||
                                                      'Unknown_Company';
                                  
                                  // Clean up employer name for filename
                                  const cleanName = employerName
                                    .replace(/[^a-zA-Z0-9\s]/g, '') // Remove special chars
                                    .replace(/\s+/g, '_') // Replace spaces with underscores
                                    .substring(0, 50); // Limit length
                                  
                                  // Try to get quarter and year
                                  const quarter = result?.quarterBeingCorrected?.quarter ||
                                                 result?.quarter_being_corrected?.quarter ||
                                                 result?.tax_period?.quarter ||
                                                 result?.quarter ||
                                                 '';
                                  
                                  const year = result?.quarterBeingCorrected?.year ||
                                              result?.quarter_being_corrected?.year ||
                                              result?.tax_period?.year ||
                                              result?.year ||
                                              '';
                                  
                                  // Get form type
                                  const formType = fileItem.detectedForm || selectedSchema;
                                  
                                  // Build filename
                                  if (quarter && year) {
                                    filename = `${cleanName}_${formType}_Q${quarter}_${year}.json`;
                                  } else if (year) {
                                    filename = `${cleanName}_${formType}_${year}.json`;
                                  } else {
                                    filename = `${cleanName}_${formType}_${new Date().toISOString().split('T')[0]}.json`;
                                  }
                                  
                                  a.download = filename;
                                  a.click();
                                }}
                              >
                                Download
                              </button>
                            </>
                          )}
                          {(fileItem.status === FILE_STATUS.FAILED || fileItem.status === FILE_STATUS.CANCELLED) && (
                            <button 
                              className="action-btn retry"
                              onClick={() => retryFile(fileItem.id)}
                              disabled={processing}
                            >
                              Retry
                            </button>
                          )}
                          {fileItem.status === FILE_STATUS.PENDING && (
                            <button 
                              className="action-btn remove"
                              onClick={() => removeFile(fileItem.id)}
                              disabled={processing}
                            >
                              Remove
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Status Bar */}
          <div className="status-bar">
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

          {/* Processing Plan */}
          {processing && currentProcessingId && (
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
                        className={`plan-step ${step.status}`}
                      >
                        <span className="step-icon">{getStepIcon(step.status)}</span>
                        <span className="step-name">{step.name}</span>
                        {step.message && (
                          <span className="step-message">- {step.message}</span>
                        )}
                      </div>
                    );
                  })
                ) : (
                  <div className="plan-loading">
                    <span className="step-icon">⏳</span>
                    <span className="step-name">Initializing processing engine...</span>
                  </div>
                )}
              </div>
              
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
          {matchResult && currentProcessingId && (
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
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PDFProcessorEnhanced;