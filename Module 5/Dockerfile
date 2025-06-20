FROM python:3.11-slim

WORKDIR /deployment/app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "deployment/app/app.py"]
