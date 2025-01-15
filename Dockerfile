# Use an official Python image as a base
FROM python:3.10-slim

LABEL authors="jakubt"

# Set the working directory to /app
WORKDIR /src

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt
RUN pip install typing_extensions

# Copy the application code
COPY . .

# Run the command to start the application
# CMD ["python3", "src/cli.py"]
