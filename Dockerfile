# Docker builds images as progressively-layered changes.
# Each time the image is built, docker will only rerun commands as necessary.
# In practice, this means that the first build takes a while, but subsequent ones are much faster, if the dockerfile is well-ordered.

FROM ubuntu:16.04

MAINTAINER Spencer Brown "sbrow420@students.kennesaw.edu"

# install necessary packages.
# This will be cached, and not rerun at every image build.
RUN apt-get update -y && \
    apt-get install -y python3-dev && apt-get install -y python3-pip

# Copy the requirements.txt into the image.
# Copy this first so that changes can be cached.
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]