#!/bin/bash
# Multi-architecture Docker build script for Smart Wheelchair WebSocket Server
# Supports: linux/amd64 (x86_64), linux/arm64, linux/arm/v7 (Raspberry Pi)

set -e

echo "=============================================="
echo "Smart Wheelchair Docker Multi-Arch Build"
echo "=============================================="
echo ""

# Check if buildx is available
if ! docker buildx version &> /dev/null; then
    echo "❌ Docker buildx not found!"
    echo "Install with: docker buildx install"
    exit 1
fi

echo "✓ Docker buildx is available"

# Check if builder exists, create if not
if ! docker buildx inspect wheelchair-builder &> /dev/null; then
    echo "Creating buildx builder: wheelchair-builder"
    docker buildx create --name wheelchair-builder --use
    docker buildx inspect --bootstrap
else
    echo "✓ Using existing builder: wheelchair-builder"
    docker buildx use wheelchair-builder
fi

echo ""
echo "Select build option:"
echo "  1) Build for current platform only (fastest)"
echo "  2) Build for x86_64 (Linux Mint, most PCs)"
echo "  3) Build for ARM64 (Raspberry Pi 4, 64-bit)"
echo "  4) Build for ARMv7 (Raspberry Pi 3, 32-bit)"
echo "  5) Build for ALL platforms (x86_64 + ARM64 + ARMv7)"
echo ""
read -p "Enter choice [1-5]: " choice

IMAGE_NAME="wheelchair-websocket:latest"
BUILD_ARGS="-f Dockerfile.websocket -t $IMAGE_NAME"

case $choice in
    1)
        echo ""
        echo "Building for current platform..."
        docker build $BUILD_ARGS .
        ;;
    2)
        echo ""
        echo "Building for x86_64 (amd64)..."
        docker buildx build --platform linux/amd64 $BUILD_ARGS --load .
        ;;
    3)
        echo ""
        echo "Building for ARM64..."
        docker buildx build --platform linux/arm64 $BUILD_ARGS --load .
        ;;
    4)
        echo ""
        echo "Building for ARMv7..."
        docker buildx build --platform linux/arm/v7 $BUILD_ARGS --load .
        ;;
    5)
        echo ""
        echo "Building for ALL platforms (x86_64 + ARM64 + ARMv7)..."
        echo "Note: This will take several minutes..."
        docker buildx build \
            --platform linux/amd64,linux/arm64,linux/arm/v7 \
            $BUILD_ARGS \
            --load \
            .
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "✅ Build complete!"
echo "=============================================="
echo ""
echo "Image: $IMAGE_NAME"
echo ""
echo "To view built images:"
echo "  docker images | grep wheelchair-websocket"
echo ""
echo "To start the server:"
echo "  docker-compose -f docker-compose.websocket.yml up -d"
echo ""
echo "Or use the quick start script:"
echo "  ./start_websocket_server.sh"
echo ""
