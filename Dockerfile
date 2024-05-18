FROM python:3.10

WORKDIR /urs/src/app
COPY . .
RUN pip install numpy && pip install torch && pip install tqdm && pip install matplotlib
CMD [ "python", "-u","./simple_tests.py"]