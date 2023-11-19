FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY web_service/requirements.txt .
# COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY web_service/app app/
# COPY app app/

# Expose the port your FastAPI application is listening on
EXPOSE 9696

# Set the command to run when the container starts
CMD ["uvicorn", "app.predict:app", "--host", "0.0.0.0", "--port", "9696"]