# SKILL: Deploy

Three deployment targets are wired in this repo. Pick the one that fits.

## 1. Local run (dev / demo laptop)

Single command starts analytics + dashboard side by side:

```powershell
.\scripts\run_all.ps1
```

This spawns two PowerShell jobs — one for the pipeline (headless), one for
Streamlit — and streams both logs into the current console.

Stop with `Ctrl+C`.

## 2. Windows single-file executable (PyInstaller)

Bundle the whole app into a single `.exe` judges can double-click:

```powershell
.\deploy\windows\build_exe.ps1
```

Output: `release/gaze-analytics.exe` (~180 MB, includes OpenVINO runtime).

Deploy by copying `release/gaze-analytics.exe` plus the `models/` folder to
the target machine. That's it — no Python install required.

## 3. Docker (Linux edge box, NUC, or CI)

```powershell
docker compose -f deploy/docker/docker-compose.yml up --build
```

- Publishes Streamlit on host port `8501`.
- Mounts `./data` and `./models` from the host so state survives restarts.
- Uses `openvino/ubuntu22_runtime:latest` as base — includes iGPU drivers.

For webcam access on Linux hosts, add `--device=/dev/video0` in the compose
file (already scaffolded, uncomment for Linux use).

## 4. Systemd services (real signage box)

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gaze-analytics gaze-dashboard
sudo journalctl -u gaze-analytics -f
```

Auto-restart on failure. Runs headless. Ships aggregates via MQTT if
`GAZE_MQTT_URL` env var is set (optional, sprint 8).

## Release tagging (triggers GitHub Actions build)

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The `build-release.yml` workflow will publish:
- `gaze-analytics-windows-x64.exe`
- `gaze-analytics:0.1.0` Docker image (to GHCR)

as a GitHub Release.
