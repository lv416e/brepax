FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy
