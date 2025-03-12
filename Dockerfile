# Use Ubuntu as base image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set up work directory for Express app
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json .
RUN npm install

# Copy Express.js application
COPY express.js .

# Copy TinyLlama model loader script
COPY load-model.sh /load-model.sh
RUN chmod +x /load-model.sh

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 3000 11434

# Run supervisor as the entry point
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
