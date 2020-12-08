# Docker builds images as progressively-layered changes.
# Each time the image is built, docker will only rerun commands as necessary.
# In practice, this means that the first build takes a while, but subsequent ones are much faster, if the dockerfile is well-ordered.

FROM ubuntu:20.04

MAINTAINER Spencer Brown "sbrow420@students.kennesaw.edu"

# install necessary packages.
# This will be cached, and not rerun at every image build.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y python3-dev && apt-get install -y python3-pip && \
    apt-get install -y python3-venv && apt-get install -y git && \
    apt-get install -yq ffmpeg libsm6 libxext6

WORKDIR /env

# setup virtual environment
ENV VIRTUAL_ENV=/env
RUN python3 -m venv $VIRTUAL_ENV

# activate virtual environment.
# We don't use env/bin/activate because docker. See https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
# copy dependencies before mounting project into container
COPY ./requirements.txt /app/requirements.txt

# make sure wheel is installed before requirements
RUN pip install wheel==0.35.1

# get dependencies with pip using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# run the application on container start
ENV FLASK_APP=flask/flaskapp/flaskapp.py
ENV FLASK_ENV=development
ENV MODEL=models/handwriting.model
ENV LABELS=models/labels.npy
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD flask run --host=0.0.0.0
