#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开机时刷新 quick_setup.ipynb，打开就能直接点训练页链接。"""
from __future__ import annotations

import json
import os
import socket
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("RVC_ROOT", "/root/YP-RVC"))
NOTEBOOK = ROOT / "quick_setup.ipynb"
PORT = int(os.environ.get("PORT", "7860"))


def fetch_text(url: str, timeout: float = 3.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""


def public_ip() -> str:
    for url in (
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://ip.sb",
        "https://icanhazip.com",
    ):
        text = fetch_text(url)
        if text and " " not in text and "." in text:
            return text.splitlines()[0].strip()
    return ""


def local_ips() -> list[str]:
    found: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except Exception:
        pass
    return found


def resolve_ip() -> str:
    ip = public_ip()
    if ip:
        return ip
    candidates = local_ips()
    return candidates[0] if candidates else "公网IP"


def build_notebook(url: str) -> dict:
    md = f"""# 有品音色训练

训练页开机后会自动启动。直接点击下面链接打开：

## [{url}]({url})

---

- 打不开时：先在控制台放行 **{PORT}**
- 华北一 C 没有独立公网 IP：用控制台域名，端口改成 `{PORT}`
- 若链接 IP 不对：把控制台里的公网 IP 填进地址即可
"""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "train-link",
                "metadata": {},
                "source": [line + "\n" for line in md.splitlines()],
            }
        ],
    }


def main() -> None:
    ip = resolve_ip()
    url = f"http://{ip}:{PORT}/"
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(
        json.dumps(build_notebook(url), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {NOTEBOOK}")
    print(f"训练页链接: {url}")


if __name__ == "__main__":
    main()
