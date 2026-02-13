####################################################
# STAGE 1: Builder
####################################################
FROM python:3.11-slim AS builder

# Set workdir
WORKDIR /app

# Configure apt to use HTTPS to avoid ISP redirect issues
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true

# Install build dependencies
# Acquire::Check-Valid-Until=false handles expired release files
RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache install
COPY requirements.txt .

# Install dependencies to a dedicated path
# --root-user-action=ignore: Suppress warning about running pip as root
# --no-cache-dir: Disable cache to avoid temp file cleanup issues
RUN pip install --upgrade pip --root-user-action=ignore && \
    pip install --prefix=/install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy app source
COPY . .

####################################################
# STAGE 2: Production Image
####################################################
FROM python:3.11-slim

# OCI Image Labels
LABEL org.opencontainers.image.title="Dental Image Classifier" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Streamlit app for dental condition classification using deep learning" \
      org.opencontainers.image.vendor="Cellula Internship" \
      org.opencontainers.image.source="https://github.com/cellula/teeth-classifier"

# Configure apt to use HTTPS to avoid ISP redirect issues
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true

# Install curl for health checks (minimal footprint)
# Acquire::Check-Valid-Until=false handles expired release files
RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set non-root user
RUN useradd --create-home appuser
USER appuser

WORKDIR /home/appuser/app

# Copy installed packages from builder
COPY --from=builder /install /home/appuser/.local

# Copy app code
COPY --from=builder /app /home/appuser/app

# Update PATH for installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Production environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Graceful shutdown signal
STOPSIGNAL SIGTERM

# Health check using curl (lighter than Python)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
 