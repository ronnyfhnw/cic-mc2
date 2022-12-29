FROM continuumio/miniconda3:latest

WORKDIR /home/cic-watcher

COPY environment.yml ./
COPY app.py ./
COPY boot.sh ./
COPY config.json ./

RUN chmod +x boot.sh

RUN conda env create -f environment.yml

RUN echo "source activate cic-watcher" > ~/.bashrc
ENV PATH /opt/conda/envs/cic-watcher/bin:$PATH

EXPOSE 5000

ENTRYPOINT ["./boot.sh"]