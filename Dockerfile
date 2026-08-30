# お出かけプランナー：Flask版（webapp.py）を動かすコンテナ。
#
# 学習済みモデル（model/）とデータ（data/）は git 管理されているので、
# ビルド時にそのままイメージへ入れる。起動のたびに学習し直す必要はない。
# モデルを作り直したいときは、リポジトリ側で python train_all.py を実行し、
# できあがった model/ ごとイメージを作り直す。

FROM python:3.11-slim

WORKDIR /app

# 依存関係だけを先に入れる。アプリのコードだけを変えたときに、
# この層（pip install の結果）を Docker のキャッシュから使い回せるようにするため。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# 予測の記録（reports/predictions.jsonl）はここに書き込む。
# 消えてほしくないときは docker-compose.yml のようにボリュームを重ねる。
RUN mkdir -p reports

ENV PORT=5000
EXPOSE 5000

# curl は入れず、標準ライブラリだけで確認する（イメージを増やさないため）。
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:5000/api/health', timeout=3)" || exit 1

CMD ["sh", "-c", "gunicorn 'webapp:create_web_app()' --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
