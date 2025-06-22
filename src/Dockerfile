FROM python:3.13-slim

WORKDIR /code

RUN apt-get update -y

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/src

# Expose the port the app runs on
EXPOSE 80

ENTRYPOINT ["fastapi", "run", "src/main.py", "--port", "80"]