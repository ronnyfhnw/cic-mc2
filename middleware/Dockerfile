FROM python:3.8

WORKDIR /middleware-app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app ./app

CMD ["python", "./app/middleware.py"]