# 1. Use the base image
FROM python:3.11.0-slim

# 2. Set the working directory
WORKDIR /app

# -------------------------------------------------------
# 3. INSTALL SYSTEM DEPENDENCIES
# -------------------------------------------------------
# pkg-config, gcc, libmysqlclient-dev -> Required for compiling MySQL drivers
# libgl1, libglib2.0-0 -> Required for OpenCV/PyQt5 (prevents "libGL.so.1" errors)
RUN apt-get update && apt-get install -y \
    pkg-config \
    gcc \
    default-libmysqlclient-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first (for Docker caching)
COPY requirements.txt .

# 5. Install Python packages
# - Upgrade pip to handle binary wheels better
# - Set timeout to 1000s to prevent "ReadTimeoutError" on slow networks
# - No cache dir to keep image size smaller
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 5b. Install Playwright browser binaries (needed for PDF/map rendering)
RUN python -m playwright install --with-deps chromium

# 6. Copy the rest of the code
COPY . .

# 7. Environment Variables
ENV FLASK_APP=app.py
# CRITICAL: Tells Qt/PyQt to run in "headless" mode (no monitor) to prevent crashes
ENV QT_QPA_PLATFORM=offscreen

# 8. Command to run the app
# Configuration for 2GB RAM:
# - workers=1: Prevents multiple copies of heavy ML libraries in RAM
# - threads=8: Allows handling multiple concurrent requests sharing the same RAM
# - timeout=0: Prevents Gunicorn from killing long-running predictions
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
