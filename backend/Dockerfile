# ── 构建阶段 : 安装依赖 ──
FROM python:3.11-slim AS builder

WORKDIR /app

# 系统依赖 (psycopg2 编译需要 libpq-dev)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖文件 (利用 Docker 层缓存: 依赖不变就不重装)
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

# ── 运行阶段 : 只带运行时依赖 ──
FROM python:3.11-slim AS runner

WORKDIR /app

# 运行时只需要 libpq (不需要 gcc)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq5 && \
    rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY data/ ./data/

# 暴露端口
EXPOSE 8000

# 环境变量默认值 (容器内)
ENV HOST=0.0.0.0
ENV PORT=8000

# 启动
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
