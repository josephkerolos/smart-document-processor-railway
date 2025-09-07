import { useEffect, useRef, useState } from 'react';

export function useWebSocket(url, onMessage, onOpen, onError, onClose) {
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 1000; // Start with 1 second

  useEffect(() => {
    if (!url) return;

    const connect = () => {
      try {
        console.log(`Attempting to connect to WebSocket: ${url}`);
        const websocket = new WebSocket(url);
        
        websocket.onopen = (event) => {
          console.log("WebSocket connected successfully");
          setIsConnected(true);
          reconnectAttemptsRef.current = 0;
          setWs(websocket);
          if (onOpen) onOpen(event);
        };

        websocket.onmessage = (event) => {
          if (onMessage) onMessage(event);
        };

        websocket.onerror = (event) => {
          console.error("WebSocket error:", event);
          if (onError) onError(event);
        };

        websocket.onclose = (event) => {
          console.log("WebSocket closed:", event.code, event.reason);
          setIsConnected(false);
          setWs(null);
          if (onClose) onClose(event);
          
          // Attempt to reconnect if not a normal closure
          if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
            reconnectAttemptsRef.current++;
            const delay = reconnectDelay * Math.pow(2, reconnectAttemptsRef.current - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              connect();
            }, delay);
          }
        };

        return websocket;
      } catch (error) {
        console.error("Failed to create WebSocket:", error);
        if (onError) onError(error);
      }
    };

    const websocket = connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close(1000, "Component unmounting");
      }
    };
  }, [url]);

  return { ws, isConnected };
}