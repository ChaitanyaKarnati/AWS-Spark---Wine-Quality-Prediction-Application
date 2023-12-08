FROM centos:7

RUN yum -y update && yum -y install python3 python3-dev python3-pip python3-virtualenv \
	java-1.8.0-openjdk wget

RUN python -V
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install numpy panda
RUN pip3 install pandas

RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz" \
&& mkdir -p /opt/spark \
&& tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
&& rm apache-spark.tgz


RUN ln -s /opt/spark-3.1.2-bin-hadoop2.7 /opt/spark
RUN (echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc && echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc && echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc)

RUN mkdir /code
RUN mkdir /code/data
RUN mkdir /code/data/csv
RUN mkdir /code/data/model
RUN mkdir /code/src
RUN mkdir /code/data/testdata.model/

COPY src/wine_test_data_prediction.py /code/src

COPY data/model/testdata.model/ /code/data/model/testdata.model
COPY data/csv/ /code/data/csv
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN /bin/bash -c "source ~/.bashrc"
RUN /bin/sh -c "source ~/.bashrc"

WORKDIR /code/

ENTRYPOINT ["/opt/spark/bin/spark-submit",  "src/wine_test_data_prediction.py"]