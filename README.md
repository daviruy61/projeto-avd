# projeto-avd

# 1) Subir os serviços base
```
docker compose up -d postgres minio mlflow fastapi jupyter
```

# 2) criando db trendz
```
docker exec -it postgres psql -U root postgres

CREATE DATABASE trendz;

CREATE USER root WITH PASSWORD 'root';

GRANT ALL PRIVILEGES ON DATABASE trendz TO root;

CREATE DATABASE thingsboard;

CREATE USER root WITH PASSWORD 'root';

GRANT ALL PRIVILEGES ON DATABASE trendz TO root;
```

# 3) Rodar instalação do ThingsBoard (schema + dados demo)
```
docker compose run --rm \
  -e INSTALL_TB=true \
  -e LOAD_DEMO=true \
  thingsboard
  ```

# 4) Subir tudo
```
docker compose up -d
```