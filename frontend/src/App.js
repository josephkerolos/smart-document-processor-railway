// App.js
import React, { useState } from "react";
import PDFProcessorEnhanced from "./PDFProcessorEnhanced_v8";
import PDFProcessorWorkflow from "./PDFProcessorWorkflow";
import PDFProcessorWorkflow_MultiFile from "./PDFProcessorWorkflow_MultiFile";
import "./App.css";
import "./ModeSelector.css";

function App() {
  const [mode, setMode] = useState("workflow-multi");

  return (
    <div className="App">
      <div className="mode-selector">
        <button 
          className={mode === "workflow-multi" ? "active" : ""}
          onClick={() => setMode("workflow-multi")}
        >
          Multi-File Workflow
        </button>
        <button 
          className={mode === "workflow" ? "active" : ""}
          onClick={() => setMode("workflow")}
        >
          Single File Workflow
        </button>
        <button 
          className={mode === "classic" ? "active" : ""}
          onClick={() => setMode("classic")}
        >
          Classic Processor
        </button>
      </div>
      
      {mode === "workflow-multi" ? (
        <PDFProcessorWorkflow_MultiFile />
      ) : mode === "workflow" ? (
        <PDFProcessorWorkflow />
      ) : (
        <PDFProcessorEnhanced />
      )}
    </div>
  );
}

export default App;
