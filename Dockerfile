FROM python:3.5

# Set environment
ENV APP_ENVIRONMENT_TYPE=production

ENV FLASK_APP_PUBLISH_PORT=8080
EXPOSE 8080

# Install dependencies
RUN apt-get update

# Make logs dir
RUN mkdir /var/log/server

# Make path available
ENV PYTHONPATH=$PYTHONPATH:/opt/ml

# Make `titanic` directory
RUN mkdir /opt/titanic

# Make temporary downloads directory (Make sure that app config uses TEMP_FILE_DIR set to this value)
RUN mkdir /tmp/s3-downloads

# Make temporary directory for titanic models
RUN mkdir /tmp/titanic

# Make directory for titanic python libs
RUN mkdir -p /opt/libs

# Run command to run flask server
CMD ["python", "/opt/titanic/server/main.py"]
