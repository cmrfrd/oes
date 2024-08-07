# ARG UBUNTU_VERSION=22.04
# FROM --platform=amd64 ubuntu:${UBUNTU_VERSION} as base
ARG CUDA_IMAGE="12.2.2-devel-ubuntu22.04"
FROM --platform=amd64 nvidia/cuda:${CUDA_IMAGE}

ARG USER=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_VERSION=21.x

## Create unpriv user
RUN groupadd --gid $USER_GID $USER \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# Install dependencies
RUN apt-get update && \
    apt-get -y install --no-install-recommends git curl wget build-essential ca-certificates default-jre pkg-config libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install python & python deps
RUN apt-get update && \
    apt-get -y install --no-install-recommends python3 python3-pip python3-setuptools python3-wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -U "huggingface_hub[cli]"
RUN python3 -m pip install openai numpy pillow cohere ipython scipy scikit-learn matplotlib pandas

# Install node
RUN curl -sL https://deb.nodesource.com/setup_${NODE_VERSION} | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install node deps
RUN npm install @openapitools/openapi-generator-cli -g

# Install rustup as root
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
ARG RUST_VERSION=nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain ${RUST_VERSION} && \
    rustup component add rust-src rust-analyzer-preview && \
    chown -R $USER:$USER /usr/local/rustup /usr/local/cargo
RUN rustup update
USER root

COPY . /oes
WORKDIR /oes
RUN cargo build --release
FROM --platform=amd64 nvidia/cuda:${CUDA_IMAGE}
COPY --from=0 /oes/target/release/oes /usr/local/bin/oes
CMD ["oes", "run"]