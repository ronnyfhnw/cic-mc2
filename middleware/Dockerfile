FROM python:3.8

WORKDIR /middleware-app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install requests

ADD ./app ./app

WORKDIR ./app

CMD ["python", "middleware.py"]