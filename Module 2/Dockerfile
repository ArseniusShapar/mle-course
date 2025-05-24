FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid $GID arsenii \
 && adduser --disabled-password --gecos "" --uid $UID --gid $GID arsenii

WORKDIR /
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefer-binary -r requirements.txt

COPY . .

RUN chown -R arsenii:arsenii /app

USER arsenii

CMD ["python", "app/main.py"]
