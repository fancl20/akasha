FROM debian:testing-slim

RUN groupadd --gid 1000 coder && \
    useradd --uid 1000 --gid 1000 -m coder

RUN apt-get update && apt-get install -y \
    curl git vim jq fish \
    nodejs npm sudo

RUN echo "coder ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/coder && \
    chmod 440 /etc/sudoers.d/coder

RUN apt-get install -y \
    python3 rustc rust-src cargo rustfmt

# Clean up
RUN rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/*

SHELL ["/usr/bin/fish"]
USER coder
