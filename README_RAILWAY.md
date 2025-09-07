# Smart Document Processor - Railway Deployment

This is a Railway-ready version of the Smart Document Processor with embedded compression functionality.

## What's Been Done

1. **Fixed Compression Functionality**
   - Added missing `compress_pdf_with_monitoring` method to FileOrganizerIntegrationV2
   - Created embedded PDF compression service that doesn't rely on external File Organizer API
   - Fallback to embedded compression when external API is unavailable

2. **Created Railway Configuration**
   - `railway.json` - Railway configuration file
   - `Dockerfile` - Multi-stage build for both frontend and backend
   - `start.sh` - Startup script for Railway deployment
   - `.env.example` - Environment variables template

3. **Embedded Compression Service**
   - Created `pdf_compression_service.py` with standalone compression using PyPDF2 and Pillow
   - No dependency on external File Organizer API
   - Automatic fallback when external service is unavailable

## Deployment to Railway

### Step 1: Prepare Your Repository

1. Initialize git repository:
```bash
cd ~/smart-document-processor-railway
git init
git add .
git commit -m "Initial commit for Railway deployment"
```

2. Create a GitHub repository and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/smart-document-processor-railway.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Railway

1. Go to [Railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your GitHub account and select the repository
5. Railway will automatically detect the Dockerfile and start deployment

### Step 3: Configure Environment Variables

In Railway dashboard, go to your service settings and add these environment variables:

**Required:**
- `GEMINI_API_KEY` - Your Google Gemini API key

**Optional:**
- `GOOGLE_DRIVE_MANAGER_URL` - Leave empty if not using Google Drive
- `FILE_ORGANIZER_URL` - Leave empty to use embedded compression
- `DEBUG_MODE` - Set to `true` for debugging

Railway will automatically set:
- `PORT` - The port Railway assigns (usually 443)
- `RAILWAY_ENVIRONMENT` - Production/staging

### Step 4: Access Your Application

Once deployed, Railway will provide you with a URL like:
`https://your-app-name.railway.app`

The application will be accessible at this URL with:
- Frontend served on the main port
- Backend API available at `/api/*` endpoints
- WebSocket support at `/ws/*` endpoints

## Features

- ✅ **Embedded Compression**: No external File Organizer API required
- ✅ **Railway-Optimized**: Dockerfile and configuration optimized for Railway
- ✅ **Automatic Restarts**: Self-healing with process monitoring
- ✅ **Health Checks**: Built-in health endpoint for Railway monitoring
- ✅ **Environment-Based Config**: All settings via environment variables

## Testing Locally

To test the application locally before deploying:

```bash
# Install Python dependencies
cd backend
pip3 install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY=your_key_here
export PORT=8080
export BACKEND_PORT=4830

# Run the application
cd ..
chmod +x start.sh
./start.sh
```

Access at http://localhost:8080

## Troubleshooting

### Compression Not Working
- The embedded compression service will automatically activate if File Organizer API is not available
- Check logs for "Successfully compressed using embedded service" message

### Application Not Starting
- Ensure GEMINI_API_KEY is set in Railway environment variables
- Check Railway logs for specific error messages

### Build Failures
- Ensure all files are committed to git
- Check that Dockerfile has correct paths

## Architecture

```
smart-document-processor-railway/
├── backend/
│   ├── main_enhanced_v12.py         # Main backend application
│   ├── file_organizer_integration_v2.py  # Fixed with compression monitoring
│   ├── pdf_compression_service.py   # NEW: Embedded compression service
│   └── requirements.txt             # Python dependencies
├── frontend/
│   ├── src/                         # React application source
│   ├── public/                      # Static assets
│   └── package.json                 # Node dependencies
├── Dockerfile                       # Multi-stage build configuration
├── start.sh                         # Startup script
├── railway.json                     # Railway configuration
└── .env.example                     # Environment variables template
```

## Support

For issues or questions about deployment:
1. Check Railway logs for detailed error messages
2. Ensure all environment variables are correctly set
3. Verify that the GitHub repository is properly connected

## Notes

- The application uses embedded compression by default, eliminating the need for external File Organizer API
- Google Drive integration is optional and can be enabled by setting GOOGLE_DRIVE_MANAGER_URL
- The frontend is built during Docker image creation for optimal performance