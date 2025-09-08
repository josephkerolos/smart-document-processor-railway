# Optimized multi-stage build for faster Railway deployments
FROM python:3.11-slim AS python-deps

# Install system dependencies and Python packages in parallel
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libpoppler-cpp-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements first (better caching)
WORKDIR /app
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Frontend builder stage (parallel)
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production --silent
COPY frontend/ ./
RUN npm run build

# Final stage - smaller image
FROM python:3.11-slim

# Copy system dependencies from python-deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from deps stage
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Create directories for processing and SQLite database
RUN mkdir -p /app/processed_documents /app/temp /app/logs /app/data

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