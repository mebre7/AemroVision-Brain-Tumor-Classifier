FROM ubuntu:latest
LABEL authors="mebra"

ENTRYPOINT ["top", "-b"]