FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv — manages Python installs
RUN curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python 3.12 and create a dedicated venv
RUN uv python install 3.12 && uv venv /opt/jester-env --python 3.12

# Install JESTER with CUDA 12 support.
# jax[cuda12] bundles its own CUDA 12 runtime wheels — no CUDA base image needed.
# blackjax is a git dependency — internet access required at build time.
RUN git clone https://github.com/nuclear-multimessenger-astronomy/jester.git /opt/jester && \
    cd /opt/jester && \
    uv pip install --python /opt/jester-env/bin/python -e ".[cuda12]"

ENV PATH="/opt/jester-env/bin:/root/.local/bin:$PATH"
ENV JAX_ENABLE_X64=1

ENTRYPOINT ["run_jester_inference"]
