#!/bin/bash
set -euo pipefail

# Multimodal Counterfactual Lab - Build Script
# This script handles building, testing, and packaging the application

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="multimodal-counterfactual-lab"
IMAGE_NAME="counterfactual-lab"
REGISTRY="${REGISTRY:-ghcr.io/terragon-labs}"
VERSION="${VERSION:-$(git describe --tags --always --dirty)}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD)

# Default values
BUILD_TYPE="development"
PUSH_IMAGE="false"
RUN_TESTS="true"
GENERATE_SBOM="false"
SECURITY_SCAN="false"
CLEAN_BUILD="false"

# Functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build script for Multimodal Counterfactual Lab

Options:
    -t, --type TYPE         Build type: development, production, security (default: development)
    -v, --version VERSION   Version tag (default: git describe)
    -p, --push              Push image to registry
    -r, --registry REGISTRY Container registry (default: ghcr.io/terragon-labs)
    --no-tests              Skip running tests
    --no-cache              Build without using cache
    --sbom                  Generate Software Bill of Materials
    --security-scan         Run security scans
    --clean                 Clean build artifacts before building
    -h, --help              Show this help message

Examples:
    $0                              # Development build
    $0 --type production --push     # Production build and push
    $0 --type security --sbom       # Security build with SBOM
    $0 --clean --no-cache           # Clean build without cache

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -p|--push)
                PUSH_IMAGE="true"
                shift
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            --no-tests)
                RUN_TESTS="false"
                shift
                ;;
            --no-cache)
                NO_CACHE="--no-cache"
                shift
                ;;
            --sbom)
                GENERATE_SBOM="true"
                shift
                ;;
            --security-scan)
                SECURITY_SCAN="true"
                shift
                ;;
            --clean)
                CLEAN_BUILD="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        error "Git is required but not installed"
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a git repository"
    fi
    
    # Check for required files
    if [[ ! -f "Dockerfile" ]]; then
        error "Dockerfile not found"
    fi
    
    if [[ ! -f "pyproject.toml" ]]; then
        error "pyproject.toml not found"
    fi
    
    success "Prerequisites check passed"
}

clean_build_artifacts() {
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        log "Cleaning build artifacts..."
        
        # Remove Python cache files
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        find . -type f -name "*.pyo" -delete 2>/dev/null || true
        
        # Remove build directories
        rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
        
        # Remove Docker build cache
        docker system prune -f --filter "label=project=${PROJECT_NAME}" || warn "Failed to prune Docker cache"
        
        success "Build artifacts cleaned"
    fi
}

run_tests() {
    if [[ "$RUN_TESTS" == "true" ]]; then
        log "Running tests..."
        
        # Install test dependencies
        pip install -e ".[test]" || error "Failed to install test dependencies"
        
        # Run linting
        log "Running code quality checks..."
        ruff check src tests || error "Linting failed"
        black --check src tests || error "Code formatting check failed"
        mypy src || warn "Type checking warnings found"
        
        # Run tests with coverage
        log "Running test suite..."
        pytest tests/ -v --cov=counterfactual_lab --cov-report=xml --cov-report=html || error "Tests failed"
        
        # Security checks
        log "Running security checks..."
        bandit -r src/ || warn "Security check warnings found"
        safety check || warn "Dependency security warnings found"
        
        success "Tests completed successfully"
    else
        warn "Skipping tests (--no-tests specified)"
    fi
}

select_dockerfile() {
    case "$BUILD_TYPE" in
        development)
            DOCKERFILE="Dockerfile"
            TARGET="builder"
            ;;
        production)
            DOCKERFILE="Dockerfile"
            TARGET="production"
            ;;
        security)
            DOCKERFILE="Dockerfile.security"
            TARGET="production"
            ;;
        *)
            error "Unknown build type: $BUILD_TYPE"
            ;;
    esac
}

build_image() {
    log "Building Docker image..."
    log "Build type: $BUILD_TYPE"
    log "Version: $VERSION"
    log "Dockerfile: $DOCKERFILE"
    log "Target: $TARGET"
    
    # Prepare build arguments
    BUILD_ARGS=(
        --build-arg "VERSION=$VERSION"
        --build-arg "BUILD_DATE=$BUILD_DATE"
        --build-arg "GIT_COMMIT=$GIT_COMMIT"
        --label "project=$PROJECT_NAME"
        --label "version=$VERSION"
        --label "build-date=$BUILD_DATE"
        --label "git-commit=$GIT_COMMIT"
        --label "build-type=$BUILD_TYPE"
    )
    
    # Add cache options
    if [[ -n "${NO_CACHE:-}" ]]; then
        BUILD_ARGS+=("$NO_CACHE")
    fi
    
    # Build the image
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    LATEST_TAG="${REGISTRY}/${IMAGE_NAME}:latest"
    
    docker build \
        "${BUILD_ARGS[@]}" \
        --target "$TARGET" \
        --file "$DOCKERFILE" \
        --tag "$FULL_IMAGE_NAME" \
        --tag "$LATEST_TAG" \
        . || error "Docker build failed"
    
    success "Image built successfully: $FULL_IMAGE_NAME"
}

generate_sbom() {
    if [[ "$GENERATE_SBOM" == "true" ]]; then
        log "Generating Software Bill of Materials (SBOM)..."
        
        # Check if syft is available
        if command -v syft &> /dev/null; then
            syft "${REGISTRY}/${IMAGE_NAME}:${VERSION}" -o spdx-json > "sbom-${VERSION}.json"
            success "SBOM generated: sbom-${VERSION}.json"
        else
            warn "syft not found, skipping SBOM generation"
        fi
    fi
}

run_security_scan() {
    if [[ "$SECURITY_SCAN" == "true" ]]; then
        log "Running security scans..."
        
        # Check if trivy is available
        if command -v trivy &> /dev/null; then
            log "Running Trivy security scan..."
            trivy image --format json --output "security-report-${VERSION}.json" "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
            
            # Check for HIGH/CRITICAL vulnerabilities
            CRITICAL_COUNT=$(cat "security-report-${VERSION}.json" | jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | length' | wc -l)
            HIGH_COUNT=$(cat "security-report-${VERSION}.json" | jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH") | length' | wc -l)
            
            if [[ $CRITICAL_COUNT -gt 0 ]]; then
                error "Found $CRITICAL_COUNT critical vulnerabilities"
            elif [[ $HIGH_COUNT -gt 5 ]]; then
                warn "Found $HIGH_COUNT high-severity vulnerabilities (threshold: 5)"
            else
                success "Security scan passed"
            fi
        else
            warn "trivy not found, skipping security scan"
        fi
        
        # Check if grype is available as alternative
        if command -v grype &> /dev/null && ! command -v trivy &> /dev/null; then
            log "Running Grype security scan..."
            grype "${REGISTRY}/${IMAGE_NAME}:${VERSION}" -o json > "grype-report-${VERSION}.json"
            success "Grype scan completed"
        fi
    fi
}

push_image() {
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        log "Pushing image to registry..."
        
        # Login check
        if ! docker info | grep -q "Username"; then
            warn "Not logged into Docker registry. Please run 'docker login $REGISTRY'"
        fi
        
        # Push image
        docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}" || error "Failed to push image"
        
        # Push latest tag for production builds
        if [[ "$BUILD_TYPE" == "production" ]]; then
            docker push "${REGISTRY}/${IMAGE_NAME}:latest" || warn "Failed to push latest tag"
        fi
        
        success "Image pushed successfully"
    fi
}

print_summary() {
    echo
    echo "=================================="
    echo "         BUILD SUMMARY"
    echo "=================================="
    echo "Project: $PROJECT_NAME"
    echo "Build Type: $BUILD_TYPE"
    echo "Version: $VERSION"
    echo "Image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    echo "Git Commit: $GIT_COMMIT" 
    echo "Build Date: $BUILD_DATE"
    
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        echo "Status: Built and pushed"
    else
        echo "Status: Built locally"
    fi
    
    echo "=================================="
    echo
    
    success "Build completed successfully!"
}

main() {
    log "Starting build process for $PROJECT_NAME"
    
    parse_args "$@"
    check_prerequisites
    clean_build_artifacts
    run_tests
    select_dockerfile
    build_image
    generate_sbom
    run_security_scan
    push_image
    print_summary
}

# Run main function with all arguments
main "$@"