import React, { useState } from "react";

const ExtractionVerification = ({ verificationData, documentType }) => {
  const [showAllDiscrepancies, setShowAllDiscrepancies] = useState(false);
  
  if (!verificationData) {
    return null;
  }

  const { extractionVerified, discrepancies = [], summary } = verificationData;
  
  // Group discrepancies by confidence
  const highConfidence = discrepancies.filter(d => d.confidence === "High");
  const mediumConfidence = discrepancies.filter(d => d.confidence === "Medium");
  const lowConfidence = discrepancies.filter(d => d.confidence === "Low" || !d.confidence);
  
  // Determine which discrepancies to show based on the toggle state
  const visibleDiscrepancies = showAllDiscrepancies 
    ? discrepancies 
    : [...highConfidence, ...mediumConfidence.slice(0, 3)];

  // Calculate if we're hiding any discrepancies
  const hiddenCount = discrepancies.length - visibleDiscrepancies.length;

  return (
    <div className="extraction-verification">
      <h3 className="section-title">Extraction Verification</h3>
      
      <div className={`verification-status ${extractionVerified ? 'verified' : 'issues'}`}>
        <div className="status-icon">
          {extractionVerified ? (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          )}
        </div>
        <div className="status-message">
          <strong>{extractionVerified ? "Extraction Verified" : "Possible Extraction Issues"}</strong>
          <p>{summary}</p>
        </div>
      </div>
      
      {visibleDiscrepancies.length > 0 && (
        <div className="discrepancies-container">
          <div className="discrepancies-header">
            <h4>
              Potential Extraction Errors
              {highConfidence.length > 0 && (
                <span className="high-confidence-badge">
                  {highConfidence.length} High Confidence
                </span>
              )}
            </h4>
            
            {discrepancies.length > 3 && (
              <button 
                className="toggle-discrepancies-btn"
                onClick={() => setShowAllDiscrepancies(!showAllDiscrepancies)}
              >
                {showAllDiscrepancies ? "Show Important Only" : `Show All (${discrepancies.length})`}
              </button>
            )}
          </div>
          
          <table className="discrepancies-table">
            <thead>
              <tr>
                <th>Confidence</th>
                <th>Type</th>
                <th>Location</th>
                <th>Document Value</th>
                <th>Extracted Value</th>
                <th>Likely Correct</th>
              </tr>
            </thead>
            <tbody>
              {visibleDiscrepancies.map((discrepancy, index) => (
                <tr key={index} className={`confidence-${discrepancy.confidence || 'low'}`}>
                  <td className={`confidence-cell ${discrepancy.confidence || 'low'}`}>
                    {discrepancy.confidence || 'Low'}
                  </td>
                  <td>{discrepancy.type}</td>
                  <td>{discrepancy.location}</td>
                  <td className="value-cell">{discrepancy.documentValue}</td>
                  <td className="value-cell error-value">{discrepancy.extractedValue}</td>
                  <td className="value-cell correction-value">{discrepancy.likelyCorrectValue}</td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {hiddenCount > 0 && !showAllDiscrepancies && (
            <div className="hidden-discrepancies-notice">
              {hiddenCount} less significant {hiddenCount === 1 ? 'issue' : 'issues'} hidden. 
              <button 
                className="show-all-link"
                onClick={() => setShowAllDiscrepancies(true)}
              >
                Show all
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ExtractionVerification;