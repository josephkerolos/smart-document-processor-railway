const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Proxy WebSocket connections first (more specific path)
  app.use(
    '/ws',
    createProxyMiddleware({
      target: 'http://localhost:4830',
      ws: true,
      changeOrigin: true,
      logLevel: 'debug', // Add debug logging
      onProxyReqWs: (proxyReq, req, socket) => {
        console.log('Proxying WebSocket request:', req.url);
      },
      onError: (err, req, res) => {
        console.error('WebSocket proxy error:', err);
      },
    })
  );
  
  // Proxy API requests
  app.use(
    '/process-documents-v3',
    createProxyMiddleware({
      target: 'http://localhost:4830',
      changeOrigin: true,
    })
  );
  
  app.use(
    '/cancel-processing',
    createProxyMiddleware({
      target: 'http://localhost:4830',
      changeOrigin: true,
    })
  );
  
  app.use(
    '/confirm-form-selection',
    createProxyMiddleware({
      target: 'http://localhost:4830',
      changeOrigin: true,
    })
  );
};