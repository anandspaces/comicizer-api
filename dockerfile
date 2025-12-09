# Dockerfile
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PIL/Pillow and reportlab
RUN apt-get update && apt-get install -y \
    curl \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pipx and uv
RUN pip install --no-cache-dir pipx && \
    pipx ensurepath && \
    pipx install uv

# Add pipx binaries to PATH
ENV PATH="/root/.local/bin:$PATH"

# Verify uv is installed
RUN uv --version

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies using uv
RUN uv pip install --system --no-cache .

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Expose port
EXPOSE 8040

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8040"]