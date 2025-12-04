import os
from datetime import datetime
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from minio import Minio
from psycopg2 import connect
from psycopg2.extras import execute_values
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient, register_model
from pydantic import BaseModel
import mlflow.pyfunc
import requests

# Configuração básica do app
app = FastAPI(
    title="Pipeline INMET - Vento",
    description="Ingestão, clusterização e persistência de dados de vento (CSV INMET) para MinIO + PostgreSQL.",
    version="1.3.0",
)

TB_URL = os.getenv("TB_URL", "http://thingsboard:8080")
TB_TOKEN_BRUTO = os.getenv("TB_TOKEN_BRUTO")
TB_TOKEN_TRATADO = os.getenv("TB_TOKEN_TRATADO")

RAW_CSV_PATH = "/data/raw/INMET_NE_PE_A341_CARUARU_01-01-2024_A_31-12-2024.CSV"
RAW_CSV_SKIPROWS = 8  # ou o valor que você já usa na ingest_wind

def send_telemetry_to_thingsboard(token: str, payload: dict):
    """
    Envia um JSON de telemetria para o dispositivo no ThingsBoard
    via API HTTP.
    """
    if not token:
        raise HTTPException(
            status_code=500,
            detail="Token do ThingsBoard não configurado (TB_TOKEN_BRUTO / TB_TOKEN_TRATADO).",
        )

    url = f"{TB_URL}/api/v1/{token}/telemetry"

    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao chamar ThingsBoard: {str(e)}",
        )

MODEL_NAME = "vento_clusters_diarios_kmeans"
MODEL_ALIAS = "Production"

class WindDayInput(BaseModel):
    u: float
    v: float
    vento_velocidade: float
    vento_rajada: float


def load_production_model():
    """
    Carrega o modelo 'Production' do Model Registry do MLflow.
    """
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        # Devolve um erro 500 se não conseguir carregar o modelo
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao carregar modelo do MLflow: {str(e)}"
        )


#configuração mlflow

# MLflow: apontar para o serviço mlflow do docker-compose
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("vento_clusters_diarios")

# Configuração MinIO
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

BUCKET_NAME = "inmet-raw"
PREFIX = "wind/raw/"

# Configuração Postgres (via variáveis de ambiente)
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")


def get_pg_connection():
    return connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_clusters_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS vento_clusters_diarios (
        data DATE PRIMARY KEY,
        u DOUBLE PRECISION,
        v DOUBLE PRECISION,
        vento_velocidade DOUBLE PRECISION,
        vento_rajada DOUBLE PRECISION,
        vento_direcao DOUBLE PRECISION,
        cluster INTEGER,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
    );
    """
    conn = get_pg_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
    finally:
        conn.close()


@app.on_event("startup")
def startup_event():
    # Garante que o bucket MinIO exista
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)

    # Garante que a tabela no Postgres exista
    ensure_clusters_table()


@app.get("/")
def root():
    return {
        "message": "API de ingestão INMET rodando.",
        "docs_url": "/docs",
        "ingest_endpoint": "/ingest/wind",
        "clusters_endpoint_json": "/clusters/daily",
        "clusters_endpoint_csv": "/clusters/daily/csv",
        "clusters_persist_endpoint": "/clusters/daily/persist",
    }


# Utilitário: lê o parquet mais recente do MinIO
def load_latest_parquet():
    objects = list(
        minio_client.list_objects(BUCKET_NAME, prefix=PREFIX, recursive=True)
    )
    if not objects:
        raise HTTPException(
            status_code=404, detail="Nenhum arquivo Parquet encontrado no MinIO."
        )

    latest_obj = sorted(objects, key=lambda o: o.object_name)[-1]

    response = minio_client.get_object(BUCKET_NAME, latest_obj.object_name)
    data = response.read()
    response.close()
    response.release_conn()

    df = pd.read_parquet(BytesIO(data))
    return df, latest_obj.object_name


# Rota de ingestão: lê CSV bruto do INMET e grava Parquet no MinIO
@app.post("/ingest/wind")
def ingest_wind():
    try:
        # Caminho do CSV montado via volume no container
        csv_path = (
            "/data/raw/INMET_NE_PE_A341_CARUARU_01-01-2024_A_31-12-2024.CSV"
        )

        # 1) Ler o arquivo como TEXTO para descobrir em qual linha está o cabeçalho real.
        with open(csv_path, "r", encoding="latin1") as f:
            lines = f.readlines()

        header_row = None
        for i, line in enumerate(lines):
            # normalmente o cabeçalho começa com "Data;" nos arquivos do INMET
            if line.startswith("Data;"):
                header_row = i
                break

        if header_row is None:
            raise HTTPException(
                status_code=400,
                detail="Não encontrei a linha de cabeçalho (linha que começa com 'Data;') no CSV.",
            )

        # 2) Ler o CSV com pandas usando essa linha como cabeçalho.
        df = pd.read_csv(
            csv_path,
            sep=";",  # INMET usa ';'
            header=header_row,  # linha do cabeçalho encontrada acima
            encoding="latin1",
        )

        # 3) Padronizar nomes das colunas (ajuste os textos conforme aparecem no seu CSV).
        df = df.rename(
            columns={
                "Data": "data",
                "Hora UTC": "hora_utc",
                "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": "vento_direcao_(gr)",
                "VENTO, RAJADA MAXIMA (m/s)": "vento_rajada_max_(m/s)",
                "VENTO, VELOCIDADE HORARIA (m/s)": "vento_velocidade_(m/s)",
            }
        )

        # 4) Conferir se as colunas necessárias existem depois do rename.
        expected_cols = [
            "data",
            "hora_utc",
            "vento_direcao_(gr)",
            "vento_rajada_max_(m/s)",
            "vento_velocidade_(m/s)",
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Colunas faltando após leitura do CSV: {missing}",
            )

        # 5) Selecionar somente as colunas que interessam.
        df = df[expected_cols]

        # 6) Salvar em Parquet na nuvem (MinIO)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"wind/raw/vento_pe_{now_str}.parquet"

        buf = BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        data_bytes = buf.getvalue()

        minio_client.put_object(
            BUCKET_NAME,
            object_name,
            BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="application/octet-stream",
        )

        # 7) Devolver uma amostra pra você ver no navegador
        sample = df.head(5).to_dict(orient="records")

        return {
            "status": "ok",
            "rows_ingested": len(df),
            "bucket": BUCKET_NAME,
            "object_name": object_name,
            "sample_preview": sample,
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="CSV de vento não encontrado."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Utilitário: calcula clusters diários e retorna df_day + resumo
def compute_daily_clusters(k: int):
    df, parquet_name = load_latest_parquet()

    # 1) Conversão robusta de tipos
    df["vento_velocidade_(m/s)"] = pd.to_numeric(
        df["vento_velocidade_(m/s)"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    df["vento_rajada_max_(m/s)"] = pd.to_numeric(
        df["vento_rajada_max_(m/s)"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    df["vento_direcao_(gr)"] = pd.to_numeric(
        df["vento_direcao_(gr)"], errors="coerce"
    )

    # 2) Vetores U/V
    df["theta_rad"] = np.deg2rad(df["vento_direcao_(gr)"])
    df["u"] = df["vento_velocidade_(m/s)"] * np.sin(df["theta_rad"])
    df["v"] = df["vento_velocidade_(m/s)"] * np.cos(df["theta_rad"])

    # 3) Agregação diária
    df_day = df.groupby("data", as_index=False).agg(
        u=("u", "mean"),
        v=("v", "mean"),
        vento_velocidade=("vento_velocidade_(m/s)", "mean"),
        vento_rajada=("vento_rajada_max_(m/s)", "mean"),
        vento_direcao=("vento_direcao_(gr)", "mean"),
    )

    df_day["data"] = pd.to_datetime(df_day["data"], format="%Y/%m/%d")
    df_day = df_day.sort_values("data").reset_index(drop=True)

    features = ["u", "v", "vento_velocidade", "vento_rajada"]
    X = df_day[features].dropna()

    # Pipeline: scaler + kmeans
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=42, n_init="auto")),
    ])

    # Treina o pipeline diretamente nos dados crus
    pipeline.fit(X)

    # Labels vêm do KMeans dentro do pipeline
    labels = pipeline.named_steps["kmeans"].labels_

    # Ajusta df_day para ter apenas as linhas válidas
    df_day = df_day.loc[X.index].copy()
    df_day["cluster"] = labels

    # Para calcular silhouette, usamos os dados escalados pelo pipeline
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    sil_score = silhouette_score(X_scaled, labels)


    # 8) Resumo por cluster
    cluster_summary = (
        df_day.groupby("cluster")
        .agg(
            num_dias=("data", "count"),
            vel_media=("vento_velocidade", "mean"),
            rajada_media=("vento_rajada", "mean"),
            direcao_media=("vento_direcao", "mean"),
            u_mean=("u", "mean"),
            v_mean=("v", "mean"),
        )
        .reset_index()
    )

    # 9) Log no MLflow
    model_name = "vento_clusters_diarios_kmeans"

    with mlflow.start_run(run_name=f"kmeans_daily_k{k}") as run:
        # --------- parâmetros ----------
        mlflow.log_param("k", k)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("parquet_usado", parquet_name)

        # --------- métricas ------------
        mlflow.log_metric("silhouette", float(sil_score))
        mlflow.log_metric("num_dias", int(len(df_day)))

        # --------- modelo --------------
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # --------- artefatos CSV -------
        tmp_day = "/tmp/vento_clusters_diarios.csv"
        tmp_sum = "/tmp/vento_clusters_summary.csv"

        df_day.to_csv(tmp_day, index=False)
        cluster_summary.to_csv(tmp_sum, index=False)

        mlflow.log_artifact(tmp_day, artifact_path="data")
        mlflow.log_artifact(tmp_sum, artifact_path="data")

        # --------- registrar modelo no Model Registry ---------
        model_uri = f"runs:/{run.info.run_id}/model"

        registered_model = register_model(
            model_uri=model_uri,
            name=model_name,
        )

        # (Opcional) marcar a versão mais recente com um alias, ex: "Production"
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=registered_model.version,
        )

        # artefatos (opcional, mas muito legal para o relatório)
        # salva df_day e summary como CSVs temporários em memória
        buf_day = StringIO()
        df_day.to_csv(buf_day, index=False)
        buf_day.seek(0)
        with open("/tmp/vento_clusters_diarios.csv", "w") as f:
            f.write(buf_day.getvalue())
        mlflow.log_artifact("/tmp/vento_clusters_diarios.csv", artifact_path="data")

        buf_sum = StringIO()
        cluster_summary.to_csv(buf_sum, index=False)
        buf_sum.seek(0)
        with open("/tmp/vento_clusters_summary.csv", "w") as f:
            f.write(buf_sum.getvalue())
        mlflow.log_artifact("/tmp/vento_clusters_summary.csv", artifact_path="data")

    return df_day, cluster_summary, parquet_name


# Rota JSON: clusters diários
@app.get("/clusters/daily")
def clusters_daily(k: int = 4):
    """
    Clusterização diária dos ventos usando K-Means.
    Retorna JSON com todos os dias + resumo por cluster.
    """
    try:
        df_day, cluster_summary, parquet_name = compute_daily_clusters(k)

        return {
            "arquivo_usado": parquet_name,
            "k": k,
            "num_dias": len(df_day),
            "clusters": df_day.to_dict(orient="records"),
            "summary": cluster_summary.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/thingsboard/raw/send")
def send_raw_csv_to_thingsboard():
    """
    Lê o CSV bruto em data/raw e envia TODAS as linhas como telemetria
    para o dispositivo de dados brutos no ThingsBoard.
    """
    if not TB_TOKEN_BRUTO:
        raise HTTPException(
            status_code=500,
            detail="TB_TOKEN_BRUTO não configurado nas variáveis de ambiente.",
        )

    try:
        # Lê o CSV cru (mas pulando as linhas até começar os dados reais)
        df = pd.read_csv(
            RAW_CSV_PATH,
            sep=";",                # ajuste se o separador for outro
            skiprows=RAW_CSV_SKIPROWS,
            encoding="latin1",      # ou "utf-8", depende do arquivo
        )

        # Remove linhas completamente vazias
        df = df.dropna(how="all")

        # Sanitiza nomes de colunas para virarem chaves de telemetria
        def sanitize_col(col: str) -> str:
            return (
                col.strip()
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
                .replace("%", "pct")
                .replace("º", "")
                .replace("°", "")
            )

        df = df.rename(columns={c: sanitize_col(c) for c in df.columns})

        sent = 0
        for _, row in df.iterrows():
            # Converte a linha em dict, tratando NaN para None
            payload = {
                "ts": 0,
                "values": {}
            }
            
            data_from_csv = None
            for k, v in row.to_dict().items():
                if pd.isna(v):
                    payload[k] = None
                else:
                    if(k == "data"):
                        data_from_csv = v
                    elif(k == "hora_utc"):
                        data_hora_str = data_from_csv + ":" + v
                        dt = datetime.strptime(data_hora_str, "%Y/%m/%d:%H%M %Z")
                        timestamp_ms = int(dt.timestamp() * 1000)
                        payload["ts"] = timestamp_ms
                    else:
                        payload["values"][k] = v

            # Envia a linha inteira como um "snapshot" de telemetria
            send_telemetry_to_thingsboard(TB_TOKEN_BRUTO, payload)
            sent += 1

        return {
            "status": "ok",
            "arquivo": RAW_CSV_PATH,
            "linhas_enviadas": sent,
        }

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo CSV bruto não encontrado em {RAW_CSV_PATH}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/thingsboard/data/send")
def send_data_postgres_to_thingsboard():
    """
    Recupera os dados tratados do PostgreSQL e os envia para o ThingsBoard.
    """
    if not TB_TOKEN_TRATADO:
        raise HTTPException(
            status_code=500,
            detail="TB_TOKEN_TRATADO não configurado nas variáveis de ambiente.",
        )

    try:
        # Conexão com o PostgreSQL
        conn = get_pg_connection()
        try:
            # Query para recuperar os dados tratados da tabela `vento_clusters_diarios`
            query = """
            SELECT data, u, v, vento_velocidade, vento_rajada, vento_direcao, cluster
            FROM vento_clusters_diarios
            ORDER BY data;
            """
            df = pd.read_sql(query, conn)

        finally:
            conn.close()

        sent = 0
        for _, row in df.iterrows():
            # Converte a linha em dict, tratando NaN para None
            payload = {
                "ts": 0,
                "values": {}
            }

            # A data precisa ser convertida em timestamp (ms desde 1970)
            timestamp_ms = int(pd.to_datetime(row['data']).timestamp() * 1000)
            payload["ts"] = timestamp_ms

            # Atribui os valores das colunas no payload
            for column in df.columns:
                if pd.notna(row[column]):
                    if column == "data": 
                        continue
                    payload["values"][column] = row[column]

            print(payload)

            # Envia para o ThingsBoard
            send_telemetry_to_thingsboard(TB_TOKEN_TRATADO, payload)
            sent += 1

        return {
            "status": "ok",
            "linhas_enviadas": sent,
            "source": "PostgreSQL",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Rota CSV: clusters diários
@app.get("/clusters/daily/csv")
def clusters_daily_csv(k: int = 4):
    """
    Clusterização diária dos ventos usando K-Means.
    Retorna um CSV para download com uma linha por dia.
    """
    try:
        df_day, cluster_summary, parquet_name = compute_daily_clusters(k)

        # Gera CSV em memória
        csv_buffer = StringIO()
        df_day.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        filename = f"vento_clusters_diarios_k{k}.csv"

        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/cluster")
def predict_cluster(payload: WindDayInput):
    """
    Usa o modelo registrado no MLflow para prever o cluster diário
    a partir de (u, v, velocidade, rajada).
    """
    # Carrega o modelo Production
    model = load_production_model()

    # Monta um DataFrame com as features na mesma ordem/nomes do treinamento
    data = pd.DataFrame([{
        "u": payload.u,
        "v": payload.v,
        "vento_velocidade": payload.vento_velocidade,
        "vento_rajada": payload.vento_rajada,
    }])

    try:
        # Como o modelo é um Pipeline (scaler + kmeans), basta chamar predict
        preds = model.predict(data)
        cluster = int(preds[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao executar predição: {str(e)}"
        )

    return {
        "input": payload.dict(),
        "cluster_previsto": cluster,
        "modelo": {
            "name": MODEL_NAME,
            "alias": MODEL_ALIAS,
        },
    }



# Rota POST: persiste clusters diários no PostgreSQL
@app.post("/clusters/daily/persist")
def clusters_daily_persist(k: int = 4):
    """
    Calcula os clusters diários (como em /clusters/daily)
    e persiste o resultado na tabela vento_clusters_diarios do PostgreSQL.
    """
    try:
        df_day, cluster_summary, parquet_name = compute_daily_clusters(k)

        # Conexão com Postgres
        conn = get_pg_connection()
        try:
            rows = []
            for _, row in df_day.iterrows():
                rows.append(
                    (
                        row["data"].date(),
                        float(row["u"]) if pd.notna(row["u"]) else None,
                        float(row["v"]) if pd.notna(row["v"]) else None,
                        float(row["vento_velocidade"])
                        if pd.notna(row["vento_velocidade"])
                        else None,
                        float(row["vento_rajada"])
                        if pd.notna(row["vento_rajada"])
                        else None,
                        float(row["vento_direcao"])
                        if pd.notna(row["vento_direcao"])
                        else None,
                        int(row["cluster"]),
                    )
                )

            sql = """
            INSERT INTO vento_clusters_diarios
                (data, u, v, vento_velocidade, vento_rajada, vento_direcao, cluster)
            VALUES %s
            ON CONFLICT (data) DO UPDATE SET
                u = EXCLUDED.u,
                v = EXCLUDED.v,
                vento_velocidade = EXCLUDED.vento_velocidade,
                vento_rajada = EXCLUDED.vento_rajada,
                vento_direcao = EXCLUDED.vento_direcao,
                cluster = EXCLUDED.cluster;
            """

            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, rows)

        finally:
            conn.close()

        return {
            "status": "ok",
            "mensagem": "Clusters diários persistidos no PostgreSQL.",
            "k": k,
            "num_dias_persistidos": len(df_day),
            "arquivo_usado": parquet_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
