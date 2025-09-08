// API Configuration for dynamic URL resolution
// Automatically detects whether we're running locally or in production

const getApiConfig = () => {
  // Get current location
  const protocol = window.location.protocol;
  const hostname = window.location.hostname;
  const port = window.location.port;
  
  // Determine API base URL based on environment
  let apiBaseUrl;
  let wsProtocol;
  
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    // Local development - use the same port as the app
    apiBaseUrl = `${protocol}//${hostname}:${port || '3000'}`;
    wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
  } else {
    // Production (Railway or other deployment) - use same origin
    apiBaseUrl = `${protocol}//${hostname}${port ? ':' + port : ''}`;
    wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
  }
  
  return {
    API_BASE_URL: apiBaseUrl,
    WS_BASE_URL: `${wsProtocol}//${hostname}${port ? ':' + port : ''}`,
    
    // API endpoints
    endpoints: {
      processDocuments: '/api/process-documents-v3',
      cancelProcessing: (sessionId) => `/api/cancel-processing/${sessionId}`,
      confirmFormSelection: '/api/confirm-form-selection',
      health: '/health',
      pngConverterStatus: '/png-converter-status',
      initiateBatch: '/api/initiate-batch',
      finalizeBatch: '/api/finalize-batch',
      batchStatus: (batchId) => `/api/batch-status/${batchId}`,
      processGdrive: '/api/process-gdrive',
      processBatchGdrive: '/api/process-batch-gdrive',
      combineResults: '/api/combine-results',
      downloadArchive: (sessionId, archiveName) => `/api/download-archive/${sessionId}/${archiveName}`,
    },
    
    // WebSocket endpoint
    websocket: (sessionId) => `${wsProtocol}//${hostname}${port ? ':' + port : ''}/ws/${sessionId}`
  };
};

// Export the configuration
const apiConfig = getApiConfig();

export default apiConfig;

// Helper function to build full API URLs
export const buildApiUrl = (endpoint) => {
  return `${apiConfig.API_BASE_URL}${endpoint}`;
};

// Helper function to build WebSocket URLs
export const buildWsUrl = (sessionId) => {
  return apiConfig.websocket(sessionId);
};