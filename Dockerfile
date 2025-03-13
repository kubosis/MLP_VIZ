# Set base image (host OS)
FROM harbor.k8ssb.cpas.cz/docker-hub/library/python:3.9.16

# Update the package list
RUN apt-get update

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Configure pip to use the custom Python package repository
RUN pip config --user set global.index-url http://nexus.cpas.cz/repository/python-group/simple  && pip config --user set global.trusted-host nexus.cpas.cz

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt

# Clone the repository containing GCP certificates
RUN git clone https://git.cpas.cz/scm/aiml/gcp-certs.git

# Copy GCP certificates to the system certificates directory
RUN cp -r gcp-certs/certs/* /etc/ssl/certs/

# Update the CA certificates
RUN update-ca-certificates --fresh

# Concatenate all certificates into a single file
RUN cat /etc/ssl/certs/* > /etc/ssl/certs/ca_certificates.crt

# Set environment variables for CA bundle and proxies
ENV REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca_certificates.crt'
ENV HTTP_PROXY='http://proxyns.cpas.cz:8080'
ENV HTTPS_PROXY='http://proxyns.cpas.cz:8080'
ENV no_proxy="localhost,127.0.0.1,.cpas.cz,.gpcz.corp.local"

# Set the working directory to /app
WORKDIR /app

# Copy the main application file into the container
COPY main.py .

# Section for users to add their own commands and code
# Users can copy their additional code and add their custom commands here

# Example:
# COPY my_code.py .
# RUN my_custom_command
# CMD [ "python", "./main.py" ]