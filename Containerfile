FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y python3.9 \
    python3-pip \
    libgmp-dev

COPY . /app
WORKDIR /app
RUN pip install .
RUN pip3 install jupyter

COPY . /

CMD ["fog_x"]
