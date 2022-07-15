# Docker Image:
## Specifying the different layers: different layers simply means that 
## one layer is indeprendent of others while building so that if we update 
## any layer it wont affect the occurence of other layer
FROM python:3.7.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app




