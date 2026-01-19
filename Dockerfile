FROM python:3.11-slim

WORKDIR /app

# Install git-lfs for large file support
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY .streamlit .streamlit

# Copy the dataset (Git LFS will handle this)
COPY NYC_crashes_dataset_STANDARDIZED.csv .

# Expose Streamlit port
EXPOSE 7860

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
