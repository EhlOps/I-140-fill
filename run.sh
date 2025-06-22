docker build src  -t chat-api
docker rm chat-api
docker run --env-file=.env -it -p 8000:80 --name chat-api chat-api