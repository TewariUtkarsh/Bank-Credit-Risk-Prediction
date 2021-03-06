FROM python:3.7.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 80
CMD gunicorn --workers=1 --bind 0.0.0.0:80 app:app



