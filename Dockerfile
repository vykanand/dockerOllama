FROM ollama/ollama:latest

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Copy your application files
COPY express.js /app/express.js
COPY test.js /app/test.js
COPY load-model.sh /app/load-model.sh

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm

# Install dependencies
WORKDIR /app
RUN npm init -y && npm install express axios

# Make the script executable
RUN chmod +x /app/load-model.sh

# Expose the port your Express app uses
EXPOSE 3000

# Start Ollama and your Express server
CMD ["/bin/bash", "-c", "/app/load-model.sh & node /app/express.js"]
