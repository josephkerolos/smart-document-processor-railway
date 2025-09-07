import React from "react";

/**
 * Component for selecting the appropriate JSON schema to use for document extraction
 * This allows the backend to tailor the extraction process to specific document types
 * 
 * Available schemas:
 * - generic: General purpose extraction for any document
 * - 1040: IRS Form 1040 U.S. Individual Income Tax Return
 * - 2848: IRS Form 2848 - Power of Attorney and Declaration of Representative
 * - 8821: IRS Form 8821 - Tax Information Authorization
 * - 941: Form 941 - Employer's QUARTERLY Federal Tax Return
 * - 941-X: Form 941-X - Adjusted Employer's QUARTERLY Federal Tax Return
 * - payroll: Universal Payroll Data Schema
 */
const SchemaSelector = ({ selectedSchema, onSchemaChange, disabled = false }) => {
  const schemas = [
    { id: "generic", name: "Generic Document" },
    { id: "1040", name: "Form 1040 - Individual Tax Return" },
    { id: "2848", name: "Form 2848 - Power of Attorney" },
    { id: "8821", name: "Form 8821 - Tax Information Authorization" },
    { id: "941", name: "Form 941 - Employer's Quarterly Tax Return" },
    { id: "941-X", name: "Form 941-X - Adjusted Quarterly Tax Return" },
    { id: "payroll", name: "Payroll Data" }
  ];

  return (
    <div className="form-group">
      <label htmlFor="schemaSelector" className="form-label">Document Type</label>
      <select
        id="schemaSelector"
        value={selectedSchema}
        onChange={(e) => onSchemaChange(e.target.value)}
        disabled={disabled}
        className="schema-selector"
        style={{
          width: '100%',
          padding: '0.75rem',
          borderRadius: '0.375rem',
          border: '1px solid #d1d5db',
          backgroundColor: disabled ? '#f3f4f6' : '#fff',
          fontSize: '1rem',
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.7 : 1,
          appearance: 'none',
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%236b7280'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'right 0.75rem center',
          backgroundSize: '1.25rem',
          paddingRight: '2.5rem'
        }}
      >
        {schemas.map(schema => (
          <option key={schema.id} value={schema.id}>
            {schema.name}
          </option>
        ))}
      </select>
      <div className="schema-info" style={{ 
        fontSize: '0.875rem', 
        marginTop: '0.5rem', 
        color: disabled ? '#9ca3af' : '#6b7280',
        fontStyle: 'italic'
      }}>
        {disabled ? 'Template locked during processing' : 'Select the document type to ensure optimal data extraction'}
      </div>
    </div>
  );
};

export default SchemaSelector;