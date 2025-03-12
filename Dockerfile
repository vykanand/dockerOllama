FROM ollama/ollama:latest

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Node.js and npm first (this rarely changes)
RUN apt-get update && apt-get install -y nodejs npm

# Set up working directory
WORKDIR /app

# Copy only package files first to leverage caching
COPY package*.json ./
RUN npm install

# Then copy the rest of the application code
# (This layer only rebuilds when your code changes)
COPY express.js .
COPY test.js .
COPY load-model.sh .

# Make the script executable
RUN chmod +x /app/load-model.sh

# Expose the port your Express app uses
EXPOSE 3000

# Start Ollama and your Express server
CMD ["/bin/bash", "-c", "/app/load-model.sh & node /app/express.js"]
