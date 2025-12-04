# Agrupamento de Padrões de Vento
Projeto de análise de dados meteorológicos para monitoramento de padrões de vento em Pernambuco.

Desenvolvido por Antonio Paulo Araujo, Clara Machado, Davi Gomes, Heloísa Tanaka, João Pedro Fontes, Larissa Sobrinho, Leonardo Cardoso - **C.E.S.A.R School**.

## Sobre o Projeto
Este projeto tem como objetivo criar uma solução automatizada para processar e analisar dados meteorológicos de vento, com foco na região de Pernambuco. Utilizando uma arquitetura conteinerizada, o sistema integra várias ferramentas modernas para coletar, armazenar, processar e visualizar os dados de vento de forma eficiente e interativa.

O resultado final é um pipeline completo que permite identificar padrões de vento (direção, velocidade e rajada) utilizando K-Means, e visualiza esses padrões em dashboards interativos, para facilitar a interpretação dos fenômenos climáticos regionais.

## Tecnologias Utilizadas
- **FastAPI**: Para ingestão e processamento dos dados.
- **MinIO**: Armazenamento de dados em objetos (arquivos .parquet).
- **PostgreSQL**: Banco de dados relacional para armazenar os resultados da análise.
- **Jupyter Notebook**: Análise exploratória de dados e modelagem.
- **MLFlow**: Para versionamento e gerenciamento de experimentos.
- **ThingsBoard**: Visualização interativa dos resultados.
- **Docker Compose**: Orquestração de containers para facilitar a execução do pipeline completo.

## Arquitetura do Sistema
A solução é construída em um ambiente conteinerizado, onde cada serviço tem seu papel específico:

1. **Ingestão de Dados**: O FastAPI lê arquivos CSV do INMET e extrai as colunas relevantes (direção, velocidade e rajada do vento).

2. **Armazenamento**: Os dados extraídos são armazenados em MinIO no formato `.parquet`, garantindo eficiência no armazenamento de grandes volumes de dados.

3. **Processamento e Modelagem**: Os dados são processados e analisados no PostgreSQL. Utilizando o algoritmo de K-Means, os dados de vento são agrupados em clusters com base em seu comportamento.

4. **Versionamento de Experimentos**: O MLFlow é utilizado para gerenciar as diferentes versões de experimentos e modelos, garantindo rastreabilidade.

5. **Visualização**: O ThingsBoard é utilizado para gerar dashboards interativos, facilitando a interpretação dos padrões de vento identificados.

## Como Funciona

O pipeline de dados funciona de forma contínua e automatizada:

1. **Coleta de Dados**: Dados meteorológicos são obtidos de arquivos CSV do INMET, com informações sobre a direção, velocidade e rajada de vento.

2. **Tratamento de Dados**: As inconsistências nos dados são tratadas e os dados brutos são limpos e estruturados.

3. **Análise de Dados**: Após a limpeza, são realizadas análises exploratórias para identificar padrões e tendências dos ventos.

4. **Clusterização com K-Means**: O algoritmo K-Means é utilizado para identificar diferentes padrões de vento (clusters), ajudando a entender as dinâmicas atmosféricas da região.

5. **Visualização Interativa**: O ThingsBoard apresenta as informações de maneira visual e interativa, com gráficos de linha, histogramas e a rosa dos ventos.

## Como Executar o Projeto

1. Subir os serviços base
   ```
   docker compose up -d postgres minio mlflow fastapi jupyter
   ```

2. Criar o database do Thingsboard
   ```
   docker exec -it postgres psql -U root postgres
    
   CREATE DATABASE thingsboard;
    
   CREATE USER root WITH PASSWORD 'root';
    
   GRANT ALL PRIVILEGES ON DATABASE thingsboard TO root;
   ```

4. Rodar instalação do Thingsboard (schema + dados)
   ```
   docker compose run --rm -e INSTALL_TB=true -e LOAD_DEMO=true thingsboard
   ```

4. Subir todos os serviços
   ```
   docker compose up -d
   ```

5. Após isso você já pode usar o projeto em cada porta respectiva

- `jupyter_app` (porta 8889)
- `mlflow` (porta 5000)
- `minio` (portas 9000 e 9001)
- `thingssboard` (porta 8080)
- `fastapi` (porta 8000)

6. Idealmente para inicializar a análise de dados rode os metodos posts e gets do fastapi e rode os notebooks Jupyter


