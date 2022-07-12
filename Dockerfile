FROM python:3.6
COPY requirements.txt .
COPY controller/ ./controller/
COPY enums/ ./enums/
COPY service/ ./service/
COPY Utils/ ./Utils/
COPY vo/ ./vo/
COPY datas/Config.py .
COPY flaskApplication.py .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
EXPOSE 5000
ENV NAME World
CMD ["python","flaskApplication.py"]
