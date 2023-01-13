FROM continuumio/miniconda3:latest

WORKDIR /home/cic-logger

COPY environment.yml ./
COPY app.py ./
COPY boot.sh ./
COPY config.json ./

RUN chmod +x boot.sh

RUN conda env create -f environment.yml

RUN echo "source activate cic-logger" > ~/.bashrc
ENV PATH /opt/conda/envs/cic-logger/bin:$PATH

EXPOSE 5000

ENTRYPOINT ["./boot.sh"]