.PHONY: setup dev build test lint clean start stop

# Default target
all: setup

# Set up development environment
setup:
	@echo "🌊 Setting up Wave development environment..."
	./wave setup

# Start development servers
dev: setup
	@echo "🌊 Starting Wave in development mode..."
	./wave start

# Start services (alias for dev)
start: dev

# Stop services
stop:
	@echo "🛑 Stopping Wave services..."
	./wave stop

# Run backend tests
test-backend:
	@echo "🧪 Running backend tests..."
	cd wave_backend && python -m pytest tests/ -v

# Run frontend tests (when available)
test-frontend:
	@echo "🧪 Running frontend tests..."
	cd wave_frontend && npm test

# Run all tests
test: test-backend test-frontend

# Lint backend code
lint-backend:
	@echo "🔍 Linting backend code..."
	cd wave_backend && ruff check . && black --check . && mypy .

# Lint frontend code
lint-frontend:
	@echo "🔍 Linting frontend code..."
	cd wave_frontend && npm run lint

# Lint all code
lint: lint-backend lint-frontend

# Format backend code
format-backend:
	@echo "✨ Formatting backend code..."
	cd wave_backend && black . && ruff check --fix .

# Format frontend code  
format-frontend:
	@echo "✨ Formatting frontend code..."
	cd wave_frontend && npm run lint:fix

# Format all code
format: format-backend format-frontend

# Build frontend for production
build-frontend:
	@echo "🏗️  Building frontend..."
	cd wave_frontend && npm run build

# Build everything
build: build-frontend

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf wave_backend/__pycache__
	rm -rf wave_backend/**/__pycache__
	rm -rf wave_backend/.pytest_cache
	rm -rf wave_backend/.mypy_cache
	rm -rf wave_frontend/dist
	rm -rf wave_frontend/node_modules/.cache
	rm -rf data/*.db
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Reset environment (clean + remove venv)
reset: clean
	@echo "🔄 Resetting environment..."
	rm -rf venv
	rm -rf wave_frontend/node_modules

# Show help
help:
	@echo "Wave Development Commands:"
	@echo "  setup          Set up development environment" 
	@echo "  dev/start      Start development servers"
	@echo "  stop           Stop services"
	@echo "  test           Run all tests"
	@echo "  lint           Lint all code"
	@echo "  format         Format all code"
	@echo "  build          Build for production"
	@echo "  clean          Clean generated files"
	@echo "  reset          Reset environment completely"