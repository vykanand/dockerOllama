FROM ollama/ollama:latest

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm

# Set up working directory
WORKDIR /app

# Copy package files first to leverage caching
COPY package*.json ./
RUN npm init -y && npm install express axios

# Copy application code
COPY express.js .
COPY test.js .
COPY load-model.sh .

# Make the script executable
RUN chmod +x /app/load-model.sh

# Expose the port your Express app uses
EXPOSE 3000

# Override the ENTRYPOINT and set a new CMD
ENTRYPOINT []
CMD ["/bin/bash", "-c", "ollama serve & /app/load-model.sh & node /app/express.js"]
