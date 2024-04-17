FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y python3.9 \
    python3-pip \
    libgmp-dev \ 
    ffmpeg

RUN pip3 install pandas \
    polars \
    numpy \
    tensorflow \
    torch \
    tensorflow_datasets \
    envlogger \
    datasets \
    pyarrow
    
COPY . /app
WORKDIR /app
RUN pip install .[full]
RUN pip3 install jupyter

COPY . /

CMD ["fog_x"]
