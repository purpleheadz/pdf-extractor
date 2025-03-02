FROM python:3.10-slim

WORKDIR /app

# タブラ関連の依存関係をインストール
RUN apt-get update && apt-get install -y \
    default-jre \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# 実行権限の付与
RUN chmod +x /app/src/extractor.py

# 作業ディレクトリを設定
WORKDIR /data

# entrypointを設定
ENTRYPOINT ["python", "/app/src/extractor.py"]

# デフォルトコマンド（上書き可能）
CMD ["--help"]