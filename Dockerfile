# Multi-stage build for Smart Document Processor
FROM node:18-alpine AS frontend-builder

# Build frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production
COPY frontend/ ./
RUN npm run build

# Python backend stage
FROM python:3.11-slim

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-cpp-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Create directories for processing
RUN mkdir -p /app/processed_documents /app/temp /app/logs

# Copy startup script
COPY start.sh ./
RUN chmod +x start.sh

# Environment variables (will be overridden by Railway)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production

# Expose port
EXPOSE $PORT

# Remove healthcheck - Railway handles this differently

# Start the application
CMD ["./start.sh"]