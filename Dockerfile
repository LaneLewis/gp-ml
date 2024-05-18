FROM python:3.10

WORKDIR /app
COPY . .
RUN pip install numpy && pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && pip install tqdm && pip install matplotlib
CMD [ "python", "-u","./simple_tests.py"]