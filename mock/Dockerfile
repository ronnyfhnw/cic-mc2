FROM python:3.8

WORKDIR /mock-app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install requests

ADD ./app ./app

EXPOSE 5501

WORKDIR ./app

CMD ["python", "mock.py"]