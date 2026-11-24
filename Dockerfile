# QS Terminal in a container.
#   docker run -it --rm ghcr.io/martoncse/quant-skills            # synthetic demo data
#   docker run -it --rm -v "$PWD/data:/work/data" ghcr.io/martoncse/quant-skills --cache-dir data
# -it is required: the terminal needs a TTY.
# Core install only. For strategies #9-#12 build with:  --build-arg EXTRAS=dl
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE* ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable ${EXTRAS:+--extra $EXTRAS} \
    && rm -rf /root/.cache

ENV PATH="/app/.venv/bin:$PATH" \
    TERM=xterm-256color \
    COLORTERM=truecolor

# The terminal looks for cached prices in ./data: mount yours at /work/data.
WORKDIR /work
ENTRYPOINT ["qs-terminal"]
CMD ["--demo"]
