#!/bin/bash
# Quick start script for Smart Wheelchair WebSocket Server

set -e

echo "=========================================="
echo "Smart Wheelchair WebSocket Server Setup"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    
    # Generate a random API secret
    API_SECRET=$(openssl rand -hex 32)
    
    # Update .env with generated secret
    sed -i "s/changeme_generate_secure_token/$API_SECRET/" .env
    
    echo "✓ Created .env file with random API_SECRET"
    echo ""
    echo "📋 Your API Secret (save this!):"
    echo "   $API_SECRET"
    echo ""
    echo "⚠️  Update this in your Flutter app at:"
    echo "   smart_wheelchair_app/lib/voice_control_page.dart"
    echo "   Line 36: static const String _apiToken = '$API_SECRET';"
    echo ""
else
    echo "✓ Found existing .env file"
    API_SECRET=$(grep "^API_SECRET=" .env | cut -d '=' -f2)
    echo ""
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "✓ Docker is available"

# Check if server is already running
if docker ps | grep -q wheelchair-websocket-server; then
    echo ""
    echo "⚠️  Server is already running!"
    echo ""
    read -p "Stop and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping server..."
        docker-compose -f docker-compose.websocket.yml down
    else
        echo "Exiting..."
        exit 0
    fi
fi

# Build Docker image
echo ""
echo "🔨 Building Docker image..."
docker build -f Dockerfile.websocket -t wheelchair-websocket:latest .

# Start server
echo ""
echo "🚀 Starting WebSocket server..."
docker-compose -f docker-compose.websocket.yml up -d

# Wait a moment for startup
sleep 2

# Check if server started successfully
if docker ps | grep -q wheelchair-websocket-server; then
    echo ""
    echo "=========================================="
    echo "✅ Server started successfully!"
    echo "=========================================="
    echo ""
    echo "Server is listening on: ws://0.0.0.0:8765"
    echo ""
    echo "🔐 Configuration:"
    echo "   API_SECRET: $API_SECRET"
    echo "   Motors: DISABLED (safe mode)"
    echo ""
    echo "📱 Update Flutter app:"
    echo "   1. Open: smart_wheelchair_app/lib/voice_control_page.dart"
    echo "   2. Line 35: Update IP address to your server"
    echo "   3. Line 36: Update token to: $API_SECRET"
    echo ""
    echo "🧪 Test with Python client:"
    echo "   pip install websockets sounddevice numpy"
    echo "   python test_websocket_client.py --record 5 --token $API_SECRET"
    echo ""
    echo "📊 View logs:"
    echo "   docker-compose -f docker-compose.websocket.yml logs -f"
    echo ""
    echo "🛑 Stop server:"
    echo "   docker-compose -f docker-compose.websocket.yml down"
    echo ""
else
    echo ""
    echo "❌ Server failed to start!"
    echo ""
    echo "View logs with:"
    echo "   docker-compose -f docker-compose.websocket.yml logs"
    exit 1
fi
