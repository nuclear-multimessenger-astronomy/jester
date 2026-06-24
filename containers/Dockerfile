FROM nvidia/cuda:12.2.0-cudnn8-runtime-rockylinux9

RUN dnf install -y python3.12 python3.12-pip git curl && dnf clean all

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"

# Install JESTER with CUDA 12 support.
# blackjax is a git dependency — internet access required at build time.
RUN git clone https://github.com/nuclear-multimessenger-astronomy/jester.git /opt/jester && \
    cd /opt/jester && \
    uv pip install --system -e ".[cuda12]"

ENV JAX_ENABLE_X64=1

ENTRYPOINT ["run_jester_inference"]
