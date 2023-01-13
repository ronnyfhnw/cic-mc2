FROM python:3.8

WORKDIR /api-app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install requests

ADD ./app ./app

WORKDIR ./app

EXPOSE 5500

CMD ["python", "main.py"]