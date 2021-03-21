FROM python:3.8

WORKDIR /usr/src/app

COPY haarcascade_frontalface_default.xml .
COPY Ngap.mp3 .
COPY NhamMat.mp3 .
COPY shape_predictor_68_face_landmarks.dat .
COPY sleep.py .
COPY requirements.txt .
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","./sleep.py"]