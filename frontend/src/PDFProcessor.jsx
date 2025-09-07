import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import VendorResearch from "./VendorResearch";
import ExtractionVerification from "./ExtractionVerification";
import SchemaSelector from "./SchemaSelector";

const PDFProcessor = () => {
  const [file, setFile] = useState(null);
  const [bankFile, setBankFile] = useState(null);
  const [message, setMessage] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [bankDragActive, setBankDragActive] = useState(false);
  const [overlayActive, setOverlayActive] = useState(false);
  const [parsedData, setParsedData] = useState(null);
  const [vendorName, setVendorName] = useState("");
  const [documentType, setDocumentType] = useState("Unknown");
  const [selectedSchema, setSelectedSchema] = useState("generic");
  const fileInputRef = useRef(null);
  const bankFileInputRef = useRef(null);
  const dropContainerRef = useRef(null);
  const bankDropContainerRef = useRef(null);

  // Handle drag events for the entire page
  useEffect(() => {
    const handleDragEnter = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setOverlayActive(true);
    };

    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };

    const handleDragLeave = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.currentTarget.contains(e.relatedTarget)) return;
      setOverlayActive(false);
    };

    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setOverlayActive(false);
      
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const droppedFile = e.dataTransfer.files[0];
        // Check if file type is valid (PDF or image)
        if (droppedFile.type.includes('pdf') || droppedFile.type.includes('image/')) {
          setFile(droppedFile);
          
          // Make the file input element reference the dropped file
          if (fileInputRef.current) {
            // Create a new DataTransfer object
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(droppedFile);
            
            // Assign the DataTransfer files to the file input
            fileInputRef.current.files = dataTransfer.files;
          }
          
          e.dataTransfer.clearData();
        } else {
          alert("Please upload a PDF or image file.");
        }
      }
    };

    // Add event listeners to the document
    document.addEventListener("dragenter", handleDragEnter);
    document.addEventListener("dragover", handleDragOver);
    document.addEventListener("dragleave", handleDragLeave);
    document.addEventListener("drop", handleDrop);

    // Clean up
    return () => {
      document.removeEventListener("dragenter", handleDragEnter);
      document.removeEventListener("dragover", handleDragOver);
      document.removeEventListener("dragleave", handleDragLeave);
      document.removeEventListener("drop", handleDrop);
    };
  }, []);

  // Handle drag events specifically for the first drop zone
  const handleDragEnterForZone = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeaveForZone = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (dropContainerRef.current && !dropContainerRef.current.contains(e.relatedTarget)) {
      setDragActive(false);
    }
  };

  const handleDropForZone = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      // Check if file type is valid (PDF or image)
      if (droppedFile.type.includes('pdf') || droppedFile.type.includes('image/')) {
        setFile(droppedFile);
        
        // Make the file input element reference the dropped file
        if (fileInputRef.current) {
          // Create a new DataTransfer object
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(droppedFile);
          
          // Assign the DataTransfer files to the file input
          fileInputRef.current.files = dataTransfer.files;
        }
        
        e.dataTransfer.clearData();
      } else {
        alert("Please upload a PDF or image file.");
      }
    }
  };

  // Handle drag events specifically for the bank statement drop zone
  const handleBankDragEnterForZone = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setBankDragActive(true);
  };

  const handleBankDragLeaveForZone = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (bankDropContainerRef.current && !bankDropContainerRef.current.contains(e.relatedTarget)) {
      setBankDragActive(false);
    }
  };

  const handleBankDropForZone = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setBankDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      // Check if file type is valid (CSV or PDF)
      if (droppedFile.type.includes('csv') || droppedFile.type.includes('pdf')) {
        setBankFile(droppedFile);
        
        // Make the file input element reference the dropped file
        if (bankFileInputRef.current) {
          // Create a new DataTransfer object
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(droppedFile);
          
          // Assign the DataTransfer files to the file input
          bankFileInputRef.current.files = dataTransfer.files;
        }
        
        e.dataTransfer.clearData();
      } else {
        alert("Please upload a CSV or PDF bank statement.");
      }
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleBankFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setBankFile(selectedFile);
    }
  };

  const handleSchemaChange = (schema) => {
    setSelectedSchema(schema);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file.");
      return;
    }
    
    // Double check the file input's files property
    if (fileInputRef.current && !fileInputRef.current.files.length && file) {
      // If somehow the file input doesn't have files but we have a file state
      // Create a new DataTransfer object
      try {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        // Assign the DataTransfer files to the file input
        fileInputRef.current.files = dataTransfer.files;
      } catch (error) {
        console.error("Error setting file input files:", error);
      }
    }
    setLoading(true);
    setMessage("");
    setDownloadUrl("");
    setParsedData(null);
    setVendorName("");
    setDocumentType("Unknown");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("schema", selectedSchema); // Add the schema to the form data

    try {
      const res = await fetch("http://localhost:4830/process-pdf", {
        method: "POST",
        body: formData,
      });
      
      if (!res.ok) {
        throw new Error(`Server responded with status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("Response:", data.response);
      
      // Parse the JSON response
      try {
        const jsonData = JSON.parse(data.response);
        setParsedData(jsonData);
        
        // Extract vendor name if available
        if (jsonData && jsonData.partyInformation && jsonData.partyInformation.vendor) {
          setVendorName(jsonData.partyInformation.vendor.name || "");
        } else if (jsonData && jsonData.documentMetadata && jsonData.documentMetadata.source) {
          setVendorName(jsonData.documentMetadata.source.name || "");
        }
        
        // Extract document type
        const docType = jsonData.documentMetadata && jsonData.documentMetadata.documentType 
          ? jsonData.documentMetadata.documentType 
          : "Unknown";
        setDocumentType(docType);
        
        // Set message based on verification results
        let statusMessage = `Document processed (${docType})`;
        
        // Check extraction verification results
        const extractionIssues = jsonData.extractionVerification && !jsonData.extractionVerification.extractionVerified;
        
        if (extractionIssues) {
          statusMessage += ". Possible data extraction issues detected.";
        } else {
          statusMessage += ". Processing successful!";
        }
        
        setMessage(statusMessage);
        
      } catch (parseError) {
        console.error("Error parsing JSON:", parseError);
      }
      
      // Use the response as plain text
      const text = data.response;
      const blob = new Blob([text], { type: "application/json" });
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
      
    } catch (error) {
      console.error("Processing error:", error);
      setMessage("An error occurred: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Placeholder for bank statement reconcile function (not functional yet)
  const handleReconcile = (e) => {
    e.preventDefault();
    alert("Bank statement reconciliation feature coming soon!");
  };

  return (
    <div className="app-container">
      {/* Full-screen drop overlay */}
      <div className={`drop-overlay ${overlayActive ? 'active' : ''}`}>
        <svg className="drop-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <div className="drop-text">Drop your file here to process</div>
      </div>
    
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
              <polyline points="14 2 14 8 20 8" />
              <path d="M8 13h2" />
              <path d="M8 17h2" />
              <path d="M14 13h2" />
              <path d="M14 17h2" />
            </svg>
            <span>DocuExtract</span>
          </div>
        </div>
      </header>

      <div className="container">
        <div className="section-title">Document Processing</div>
        <h1>Extract & Reconcile Financial Document Data</h1>
        <p className="description">
          Upload your financial documents to extract structured information, then reconcile with your bank statements for complete financial verification.
        </p>

        <div className="two-column-layout">
          {/* LEFT COLUMN - DOCUMENT UPLOAD */}
          <div className="card">
            <div className="section-title">1. Document Extraction</div>
            <h2>Upload Document</h2>
            <p className="description">
              Upload an invoice, receipt, or financial document to extract structured data.
            </p>

            <form onSubmit={handleSubmit}>
              {/* Schema selector added here */}
              <SchemaSelector
                selectedSchema={selectedSchema}
                onSchemaChange={handleSchemaChange}
              />

              <div className="form-group">
                <label htmlFor="pdfFile" className="form-label">Upload Document</label>
                <div 
                  ref={dropContainerRef}
                  className={`file-input-container ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleDragEnterForZone}
                  onDragOver={(e) => e.preventDefault()}
                  onDragLeave={handleDragLeaveForZone}
                  onDrop={handleDropForZone}
                >
                  <div className="pulse-ring"></div>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <div className="file-input-text">
                    <div className="file-input-title">Choose a document or drag and drop</div>
                    <div className="file-input-description">PDF, JPG, PNG (max. 10MB)</div>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    id="pdfFile"
                    accept="application/pdf, image/*"
                    className="file-input"
                    onChange={handleFileChange}
                    required={!file} // Only required if no file is selected
                  />
                </div>
                {file && (
                  <div className="file-name">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="22 4 12 14.01 9 11.01" />
                    </svg>
                    <span>{file.name}</span>
                  </div>
                )}
              </div>

              <button 
                type="submit" 
                className="btn" 
                disabled={loading || !file}>
                {loading ? (
                  <>
                    <div className="spinner-button">
                      <svg className="spinner-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="2" x2="12" y2="6" />
                        <line x1="12" y1="18" x2="12" y2="22" />
                        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
                        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
                        <line x1="2" y1="12" x2="6" y2="12" />
                        <line x1="18" y1="12" x2="22" y2="12" />
                        <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
                        <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
                      </svg>
                      Processing Document...
                    </div>
                  </>
                ) : (
                  <>
                    <svg className="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="23 3 12 14 9 11" />
                    </svg>
                    Extract Data
                  </>
                )}
              </button>
            </form>

            {/* Result container section after successful processing */}
            {(message || downloadUrl) && (
              <div className="result-container">
                {message && (
                  <div className="result-message">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="22 4 12 14.01 9 11.01" />
                    </svg>
                    {message}
                  </div>
                )}

                {downloadUrl && (
                  <a href={downloadUrl} download="document_data.json" className="download-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Download Structured Data
                  </a>
                )}
                
                {/* Add Extraction Verification component if we have parsed data */}
                {parsedData && parsedData.extractionVerification && (
                  <ExtractionVerification 
                    verificationData={parsedData.extractionVerification}
                    documentType={documentType}
                  />
                )}
              </div>
            )}
          </div>

          {/* RIGHT COLUMN - BANK STATEMENT */}
          <div className="card">
            <div className="section-title">2. Reconciliation</div>
            <h2>Bank Statement</h2>
            <p className="description">
              Upload your bank statement to reconcile against the extracted financial data.
            </p>

            <form onSubmit={handleReconcile}>
              <div className="form-group">
                <label htmlFor="bankFile" className="form-label">Upload Bank Statement</label>
                <div 
                  ref={bankDropContainerRef}
                  className={`file-input-container ${bankDragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleBankDragEnterForZone}
                  onDragOver={(e) => e.preventDefault()}
                  onDragLeave={handleBankDragLeaveForZone}
                  onDrop={handleBankDropForZone}
                >
                  <div className="pulse-ring"></div>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="3" y1="9" x2="21" y2="9"></line>
                    <line x1="9" y1="21" x2="9" y2="9"></line>
                  </svg>
                  <div className="file-input-text">
                    <div className="file-input-title">Choose a bank statement or drag and drop</div>
                    <div className="file-input-description">CSV, PDF (max. 10MB)</div>
                  </div>
                  <input
                    ref={bankFileInputRef}
                    type="file"
                    id="bankFile"
                    accept=".csv, application/pdf"
                    className="file-input"
                    onChange={handleBankFileChange}
                  />
                </div>
                {bankFile && (
                  <div className="file-name">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="22 4 12 14.01 9 11.01" />
                    </svg>
                    <span>{bankFile.name}</span>
                  </div>
                )}
              </div>

              <button 
                type="submit" 
                className="btn bank-btn" 
                disabled={!bankFile || !downloadUrl}>
                <svg className="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                  <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                  <path d="M9 14l2 2 4-4"></path>
                </svg>
                Reconcile Transactions
              </button>
              <div className="coming-soon-badge">Coming Soon</div>
            </form>
          </div>
        </div>
        
        {/* Vendor Research section */}
        {parsedData && vendorName && (
          <VendorResearch 
            vendorName={vendorName}
            jsonData={parsedData}
          />
        )}
      </div>

      <footer className="footer">
        <p>© {new Date().getFullYear()} DocuExtract. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default PDFProcessor;