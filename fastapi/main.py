from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from datetime import datetime
from io import BytesIO, StringIO

import numpy as np
from minio import Minio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# Configuração básica do app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Pipeline INMET - Vento",
    description="Ingestão e clusterização de dados de vento (CSV INMET) para MinIO.",
    version="1.2.0"
)

# Configuração do cliente MinIO (assumindo serviço "minio" no docker-compose)
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

BUCKET_NAME = "inmet-raw"
PREFIX = "wind/raw/"


@app.on_event("startup")
def startup_event():
    # Garante que o bucket exista
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)


@app.get("/")
def root():
    return {
        "message": "API de ingestão INMET rodando.",
        "docs_url": "/docs",
        "ingest_endpoint": "/ingest/wind",
        "clusters_endpoint_json": "/clusters/daily",
        "clusters_endpoint_csv": "/clusters/daily/csv",
    }


# ---------------------------------------------------------------------
# Função utilitária: lê o parquet mais recente do MinIO
# ---------------------------------------------------------------------
def load_latest_parquet():
    objects = list(minio_client.list_objects(BUCKET_NAME, prefix=PREFIX, recursive=True))
    if not objects:
        raise HTTPException(status_code=404, detail="Nenhum arquivo Parquet encontrado no MinIO.")

    latest_obj = sorted(objects, key=lambda o: o.object_name)[-1]

    response = minio_client.get_object(BUCKET_NAME, latest_obj.object_name)
    data = response.read()
    response.close()
    response.release_conn()

    df = pd.read_parquet(BytesIO(data))
    return df, latest_obj.object_name


# ---------------------------------------------------------------------
# Rota de ingestão: lê CSV bruto do INMET e grava Parquet no MinIO
# ---------------------------------------------------------------------
@app.post("/ingest/wind")
def ingest_wind():
    try:
        # Caminho do CSV montado via volume no container
        csv_path = "/data/raw/INMET_NE_PE_A341_CARUARU_01-01-2024_A_31-12-2024.CSV"

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
                detail="Não encontrei a linha de cabeçalho (linha que começa com 'Data;') no CSV."
            )

        # 2) Agora sim, ler o CSV com pandas usando essa linha como cabeçalho.
        df = pd.read_csv(
            csv_path,
            sep=';',               # INMET usa ';'
            header=header_row,     # linha do cabeçalho encontrada acima
            encoding='latin1'
        )

        # 3) Padronizar nomes das colunas (ajuste os textos conforme aparecem no seu CSV).
        df = df.rename(columns={
            'Data': 'data',
            'Hora UTC': 'hora_utc',
            'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'vento_direcao_(gr)',
            'VENTO, RAJADA MAXIMA (m/s)': 'vento_rajada_max_(m/s)',
            'VENTO, VELOCIDADE HORARIA (m/s)': 'vento_velocidade_(m/s)'
        })

        # 4) Conferir se as colunas necessárias existem depois do rename.
        expected_cols = [
            'data',
            'hora_utc',
            'vento_direcao_(gr)',
            'vento_rajada_max_(m/s)',
            'vento_velocidade_(m/s)'
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Colunas faltando após leitura do CSV: {missing}"
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
            content_type="application/octet-stream"
        )

        # 7) Devolver uma amostra pra você ver no navegador
        sample = df.head(5).to_dict(orient="records")

        return {
            "status": "ok",
            "rows_ingested": len(df),
            "bucket": BUCKET_NAME,
            "object_name": object_name,
            "sample_preview": sample
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV de vento não encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# Função utilitária: calcula clusters diários e retorna df_day + resumo
# ---------------------------------------------------------------------
def compute_daily_clusters(k: int):
    df, parquet_name = load_latest_parquet()

    # 1) Conversão robusta de tipos
    df["vento_velocidade_(m/s)"] = pd.to_numeric(
        df["vento_velocidade_(m/s)"].astype(str).str.replace(",", "."),
        errors="coerce"
    )
    df["vento_rajada_max_(m/s)"] = pd.to_numeric(
        df["vento_rajada_max_(m/s)"].astype(str).str.replace(",", "."),
        errors="coerce"
    )
    df["vento_direcao_(gr)"] = pd.to_numeric(
        df["vento_direcao_(gr)"],
        errors="coerce"
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
        vento_direcao=("vento_direcao_(gr)", "mean")
    )

    df_day["data"] = pd.to_datetime(df_day["data"], format="%Y/%m/%d")
    df_day = df_day.sort_values("data").reset_index(drop=True)

    # 4) Seleção das features
    features = ["u", "v", "vento_velocidade", "vento_rajada"]
    X = df_day[features]

    # Remover linhas inválidas
    mask = X.notna().all(axis=1)
    df_day = df_day[mask].reset_index(drop=True)
    X = X[mask]

    if len(df_day) < k:
        raise HTTPException(
            status_code=400,
            detail=f"Número de dias válidos ({len(df_day)}) é menor que k={k}."
        )

    # 5) Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6) K-Means
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(X_scaled)
    df_day["cluster"] = labels

    # 7) Resumo por cluster
    cluster_summary = (
        df_day.groupby("cluster")
        .agg(
            num_dias=("data", "count"),
            vel_media=("vento_velocidade", "mean"),
            rajada_media=("vento_rajada", "mean"),
            direcao_media=("vento_direcao", "mean"),
            u_mean=("u", "mean"),
            v_mean=("v", "mean")
        )
        .reset_index()
    )

    return df_day, cluster_summary, parquet_name


# ---------------------------------------------------------------------
# Rota JSON: clusters diários
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Rota CSV: clusters diários
# ---------------------------------------------------------------------
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
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
