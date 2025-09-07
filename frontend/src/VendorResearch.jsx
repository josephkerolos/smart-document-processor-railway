import React, { useState } from "react";
import "./App.css";

const VendorResearch = ({ vendorName, jsonData }) => {
  const [vendorInfo, setVendorInfo] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [transactionPurpose, setTransactionPurpose] = useState("");
  const [categorization, setCategorization] = useState(null);
  const [categorizationLoading, setCategorizationLoading] = useState(false);
  const [categorizationError, setCategorizationError] = useState("");

  const researchVendor = async () => {
    if (!vendorName) return;
    
    setLoading(true);
    setError("");
    setVendorInfo("");
    setCategorization(null);
    
    try {
      console.log("Sending request to research vendor:", vendorName);
      
      const response = await fetch("http://localhost:4830/research-vendor", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ vendor_name: vendorName }),
      });
      
      console.log("Received response:", response);
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Response data:", data);
      
      if (data.error) {
        setError(data.error);
        return;
      }
      
      if (!data.response) {
        setError("Invalid response from server");
        return;
      }
      
      // Simply use the text response directly
      setVendorInfo(data.response);
      
      // After getting vendor info, categorize the transaction
      if (jsonData) {
        categorizeTransaction(data.response);
      }
      
    } catch (err) {
      console.error("Error in vendor research:", err);
      setError(`Failed to research vendor: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const categorizeTransaction = async (vendorDetails) => {
    if (!vendorDetails || !jsonData) return;
    
    setCategorizationLoading(true);
    setCategorizationError("");
    setCategorization(null);
    
    try {
      console.log("Sending request to categorize transaction");
      
      const response = await fetch("http://localhost:4830/categorize-transaction", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          vendor_info: vendorDetails,
          document_data: jsonData,
          transaction_purpose: transactionPurpose
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        setCategorizationError(data.error);
        return;
      }
      
      setCategorization(data.response);
      
    } catch (err) {
      console.error("Error in transaction categorization:", err);
      setCategorizationError(`Failed to categorize transaction: ${err.message}`);
    } finally {
      setCategorizationLoading(false);
    }
  };

  // Helper function to convert plain text with line breaks to formatted HTML
  const formatTextWithBreaks = (text) => {
    // If the text is empty, return nothing
    if (!text) return null;
    
    // Split text by line breaks and create paragraphs
    const paragraphs = text.split(/\n\n+/);
    
    return (
      <>
        {paragraphs.map((paragraph, index) => {
          // Check if this paragraph looks like a heading (short and ends with a colon)
          const isHeading = paragraph.length < 50 && paragraph.trim().endsWith(':');
          
          if (isHeading) {
            return <h4 key={index}>{paragraph}</h4>;
          }
          
          // For regular paragraphs, handle internal line breaks
          const lines = paragraph.split(/\n/);
          
          return (
            <p key={index}>
              {lines.map((line, lineIndex) => (
                <React.Fragment key={lineIndex}>
                  {line}
                  {lineIndex < lines.length - 1 && <br />}
                </React.Fragment>
              ))}
            </p>
          );
        })}
      </>
    );
  };

  return (
    <div className="vendor-research">
      <h3 className="section-title">Vendor Information</h3>
      
      <div className="vendor-details">
        <p>
          <strong>Vendor Name:</strong> {vendorName || "Not available"}
        </p>
        
        {jsonData && jsonData.partyInformation && jsonData.partyInformation.vendor && (
          <>
            <p>
              <strong>Address:</strong> {jsonData.partyInformation.vendor.address || "Not available"}
            </p>
            <p>
              <strong>Contact:</strong> {jsonData.partyInformation.vendor.contact || "Not available"}
            </p>
            <p>
              <strong>Tax ID:</strong> {jsonData.partyInformation.vendor.taxID || "Not available"}
            </p>
          </>
        )}
      </div>
      
      {/* Transaction Purpose Input */}
      <div className="form-group">
        <label htmlFor="transactionPurpose" className="form-label">Transaction Purpose (Optional)</label>
        <textarea
          id="transactionPurpose"
          value={transactionPurpose}
          onChange={(e) => setTransactionPurpose(e.target.value)}
          placeholder="Describe what this invoice is for (e.g., 'gym equipment', 'office supplies', 'consulting services'). This helps with accurate categorization."
          className="form-control"
          rows="3"
          style={{
            width: '100%',
            padding: '0.75rem',
            borderRadius: '0.375rem',
            border: '1px solid #d1d5db',
            fontSize: '1rem',
            marginBottom: '1rem',
            fontFamily: 'inherit',
            resize: 'vertical'
          }}
        />
      </div>
      
      <div className="info-message" style={{
        padding: '0.75rem',
        backgroundColor: '#f0f9ff',
        border: '1px solid #bae6fd',
        borderRadius: '0.375rem',
        marginBottom: '1rem',
        fontSize: '0.875rem',
        color: '#0369a1'
      }}>
        <strong>Note:</strong> The transaction will be categorized from your perspective as the invoice recipient, 
        not from the vendor's perspective. For example, purchases from vendors are typically expenses for your business.
      </div>
      
      <button 
        onClick={researchVendor} 
        className="btn research-btn"
        disabled={loading || !vendorName}
      >
        {loading ? (
          <span className="spinner-button">
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
            Researching...
          </span>
        ) : (
          <>
            <svg className="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            Research & Categorize
          </>
        )}
      </button>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      {vendorInfo && (
        <div className="vendor-info-wrapper">
          <div className="vendor-info">
            {formatTextWithBreaks(vendorInfo)}
          </div>
        </div>
      )}
      
      {/* Financial Categorization Results */}
      {categorizationLoading && (
        <div className="financial-categorization">
          <h3 className="section-title">Financial Categorization</h3>
          <div className="loading-indicator">
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
            Categorizing transaction...
          </div>
        </div>
      )}
      
      {categorizationError && (
        <div className="financial-categorization">
          <h3 className="section-title">Financial Categorization</h3>
          <div className="error-message">
            {categorizationError}
          </div>
        </div>
      )}
      
      {categorization && (
        <div className="financial-categorization">
          <h3 className="section-title">Financial Categorization (From Recipient's Perspective)</h3>
          <div className="categorization-card">
            <div className="categorization-header">
              <span className="category-name">{categorization.category} - {categorization.subcategory}</span>
              <span className="category-tag" data-type={categorization.ledgerType}>
                {categorization.ledgerType}
              </span>
            </div>
            <div className="categorization-details">
              <p><strong>Company:</strong> {categorization.companyName}</p>
              <p><strong>Description:</strong> {categorization.description}</p>
              
              {/* Add the explanation section */}
              {categorization.explanation && (
                <div className="explanation-section">
                  <h4>Explanation</h4>
                  <div className="explanation-content">
                    {categorization.explanation.split('\n').map((paragraph, i) => (
                      <p key={i}>{paragraph}</p>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Add debug info if needed */}
              <div className="debug-info">
                <details>
                  <summary>Debug Info</summary>
                  <pre>{JSON.stringify(categorization, null, 2)}</pre>
                </details>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VendorResearch;