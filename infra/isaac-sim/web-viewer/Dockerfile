FROM nvcr.io/nvidia/base/ubuntu:noble-20251013 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && curl -fsSL --proto '=https' https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN echo '@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/' >> /root/.npmrc \
    && npx @nvidia/create-ov-web-rtc-app@2.1.5 --name=web-viewer --sample=local-sample --external --outputDir=/app sample-app --framework=react \
    && rm -f /root/.npmrc

WORKDIR /app/web-viewer

ARG ISAACSIM_SIGNAL_HOST=127.0.0.1
ARG ISAACSIM_MEDIA_HOST=127.0.0.1
ARG ISAACSIM_SIGNAL_PORT=49100
ARG ISAACSIM_STREAM_PORT=47998

RUN grep -qF "signalingServer: '127.0.0.1'," src/main.tsx \
    && sed -i "s/signalingServer: [^,]*/signalingServer: '${ISAACSIM_SIGNAL_HOST}'/" src/main.tsx \
    && sed -i "/signalingServer:/a\\    signalingPort: ${ISAACSIM_SIGNAL_PORT},\n    mediaServer: '${ISAACSIM_MEDIA_HOST}',\n    mediaPort: ${ISAACSIM_STREAM_PORT},\n    forceWSS: false," src/main.tsx \
    && grep -qF "signalingPort: ${ISAACSIM_SIGNAL_PORT}," src/main.tsx \
    && grep -qF "mediaPort: ${ISAACSIM_STREAM_PORT}," src/main.tsx

RUN npm install --ignore-scripts \
    && npm run build

FROM nvcr.io/nvidia/base/ubuntu:noble-20251013

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && curl -fsSL --proto '=https' https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser --gid 1234 \
    && useradd --no-log-init -r -g appuser --uid 1234 -d /app -s /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chmod=755 /app/web-viewer/package.json ./

RUN echo '@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/' >> /root/.npmrc \
    && npm install --ignore-scripts \
    && npm cache clean --force \
    && rm -f /root/.npmrc \
    && chown -R appuser:appuser node_modules

COPY --from=builder --chmod=755 /app/web-viewer/dist ./dist

USER appuser

EXPOSE 8210

CMD ["./node_modules/.bin/vite", "preview", "--host", "--port", "8210"]
