FROM ubuntu:22.04

# non interactive frontend for locales
ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKER_BUILDKIT=0

# installing texlive and utils
RUN apt update \
    && apt install -y texlive texlive-lang-german texlive-latex-extra \
    && apt install -y locales \
    && apt install -y python3-pygments \
    && apt install -y git-all \
    && rm -rf /var/lib/apt/lists/*

# set locales
ENV LANG de_DE.UTF-8
ENV LANGUAGE de_DE.UTF-8
ENV LC_ALL de_DE.UTF-8

