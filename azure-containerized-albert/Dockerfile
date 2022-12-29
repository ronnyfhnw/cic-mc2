FROM python:3.8

WORKDIR /api-app

COPY requirements.txt .

RUN pip install -r requirements.txt

ADD ./app ./app

WORKDIR ./app

CMD ["python", "main.py"]