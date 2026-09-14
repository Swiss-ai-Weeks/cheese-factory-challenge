FROM mcr.microsoft.com/playwright:v1.63.0-noble

ARG WEB_VIEWER_PORT=8210
ENV DEBIAN_FRONTEND=noninteractive
ENV WEB_VIEWER_PORT=${WEB_VIEWER_PORT}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl novnc openbox websockify x11vnc xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=755 start.sh /usr/local/bin/start-isaac-desktop
COPY index.html /usr/share/novnc/index.html

EXPOSE 6080

CMD ["/usr/local/bin/start-isaac-desktop"]
