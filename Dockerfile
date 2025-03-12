FROM ollama/ollama:latest

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Node.js and npm
# Install Node.js 20.x
RUN apt-get update && apt-get install -y ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs

# Set up working directory
WORKDIR /app

# Copy package files first to leverage caching
COPY package*.json ./
RUN npm init -y && npm install express axios

# Copy application code
COPY express.js .
COPY test.js .
COPY load-model.sh .

# Add this before running the application
RUN cat /app/express.js | head -5

# Make the script executable
RUN chmod +x /app/load-model.sh

# Expose the port your Express app uses
EXPOSE 3000

# Override the ENTRYPOINT and set a new CMD
ENTRYPOINT []
CMD ["/bin/bash", "-c", "ollama serve & /app/load-model.sh & node /app/express.js"]
