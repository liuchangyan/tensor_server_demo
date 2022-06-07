#FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/classic-watchdog:0.2.0 as watchdog
#FROM --platform=${TARGETPLATFORM:-linux/amd64} python:3-alpine
FROM --platform=linux/x86_64 python:3.9
#FROM --platform=linux/amd64 tensorflow/tensorflow:latest

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE

#COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
#RUN chmod +x /usr/bin/fwatchdog
#RUN apk --no-cache add ca-certificates ${ADDITIONAL_PACKAGE}


# Add non root user
RUN groupadd -r app && useradd -r -g app app
#RUN addgroup --system app
#RUN adduser app --system --gid app

WORKDIR /home/app/
RUN mkdir -p faster_rcnn_inception_resnet_v2_640x640_1
COPY test_server.py           .
COPY a.webp  .
COPY requirements.txt   .
COPY faster_rcnn_inception_resnet_v2_640x640_1 /home/app/faster_rcnn_inception_resnet_v2_640x640_1
RUN chown -R app /home/app && \
  mkdir -p /home/app/python && chown -R app /home/app
USER app
ENV PATH=$PATH:/home/app/.local/bin:/home/app/python/bin/
ENV PYTHONPATH=$PYTHONPATH:/home/app/python
RUN /usr/local/bin/python3 -m pip install --upgrade pip

ENV PYTHONPATH=$PYTHONPATH:/home/app/python
#RUN python3 -m pip install --force-reinstall https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp38-cp38-manylinux2010_x86_64.whl
#RUN python3 -m pip install --force-reinstall https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp38-cp38-manylinux2010_x86_64.whl
#RUN python3 -m pip install --force-reinstall https://tf.novaal.de/barcelona/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl
RUN pip3 install -r requirements.txt --target=/home/app/python


WORKDIR /home/app/

USER root

# Allow any user-id for OpenShift users.
RUN chown -R app:app ./ && \
  chmod -R 777 /home/app/python

USER app

#ENV FFFPRO="python3 index.py"
#ENV fprocess="python3 index.py"
EXPOSE 9002

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

#CMD ["fwatchdog"]
ENTRYPOINT ["python3","test_server.py"]

