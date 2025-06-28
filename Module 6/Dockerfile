FROM python:3.11-slim

WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV PYTHONUNBUFFERED=TRUE

RUN echo '#!/usr/bin/env bash\npython app.py' > /usr/local/bin/serve \
 && chmod +x /usr/local/bin/serve

ENTRYPOINT ["serve"]
