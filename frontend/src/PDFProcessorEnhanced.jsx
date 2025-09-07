import React, { useState, useRef, useEffect } from "react";
import "./PDFProcessorEnhanced.css";
import SchemaSelector from "./SchemaSelector";

const PDFProcessorEnhanced = () => {
  const [files, setFiles] = useState([]);
  const [selectedSchema, setSelectedSchema] = useState("generic");
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [processingSteps, setProcessingSteps] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [ws, setWs] = useState(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const websocket = new WebSocket("ws://localhost:4830/ws");
    
    websocket.onopen = () => {
      console.log("WebSocket connected");
      setWs(websocket);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "step_update") {
        setProcessingSteps(prev => [...prev, data.step]);
      }
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close();
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
    setProcessingSteps([]);
    setResults([]);

    const formData = new FormData();
    files.forEach(file => {
      formData.append("files", file);
    });
    formData.append("schema", selectedSchema);

    try {
      const response = await fetch("http://localhost:4830/process-documents", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error("Processing error:", error);
      alert(`Error processing documents: ${error.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const formatDuration = (duration) => {
    if (!duration) return "";
    return `${duration.toFixed(2)}s`;
  };

  const getStepIcon = (status) => {
    switch (status) {
      case "started":
        return "⏳";
      case "completed":
        return "✅";
      case "failed":
        return "❌";
      case "warning":
        return "⚠️";
      default:
        return "📋";
    }
  };

  const getValidationSummary = (validation) => {
    if (!validation) return null;
    
    const percentage = 100 - validation.empty_percentage;
    const color = percentage > 80 ? "#10b981" : percentage > 50 ? "#f59e0b" : "#ef4444";
    
    return (
      <div className="validation-summary">
        <div className="validation-bar">
          <div 
            className="validation-fill" 
            style={{ width: `${percentage}%`, backgroundColor: color }}
          />
        </div>
        <div className="validation-text">
          {percentage.toFixed(1)}% fields populated ({validation.populated_count}/{validation.total_fields})
        </div>
      </div>
    );
  };

  return (
    <div className="pdf-processor-enhanced">
      <div className="processor-header">
        <h1>Document Processing System</h1>
        <p className="subtitle">AI-powered document extraction with real-time processing insights</p>
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
            className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDragIn}
            onDragLeave={handleDragOut}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
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

          {/* Process Button */}
          <button
            className="process-button"
            onClick={processDocuments}
            disabled={processing || files.length === 0}
          >
            {processing ? (
              <>
                <span className="spinner">⟳</span> Processing...
              </>
            ) : (
              <>📊 Process Documents</>
            )}
          </button>
        </div>

        <div className="right-panel">
          {/* Processing Steps */}
          {processingSteps.length > 0 && (
            <div className="processing-steps">
              <h3>Processing Progress</h3>
              <div className="steps-container">
                {processingSteps.slice(-10).map((step, index) => (
                  <div key={index} className={`step-item ${step.status}`}>
                    <div className="step-header">
                      <span className="step-message">{step.message}</span>
                      {step.duration && (
                        <span className="step-duration">{formatDuration(step.duration)}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results */}
          {results.length > 0 && (
            <div className="results-section">
              <h3>Extraction Results</h3>
              {results.map((result, index) => (
                <div key={index} className="result-item">
                  <div className="result-header">
                    <h4>Document {index + 1}</h4>
                    {result._metadata && (
                      <span className="result-meta">
                        {result._metadata.detected_form_type || result._metadata.selected_schema} 
                        • Pages {result._metadata.pages}
                        • {formatDuration(result._metadata.processing_time)}
                      </span>
                    )}
                  </div>
                  
                  {result._metadata?.validation && (
                    getValidationSummary(result._metadata.validation)
                  )}
                  
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
                      Download JSON
                    </button>
                  </div>
                  
                  {result.error && (
                    <div className="error-message">
                      Error: {result.error}
                    </div>
                  )}
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