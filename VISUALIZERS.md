# Visualization Tools

Three standalone visualization tools for analyzing and tuning controllers in the comma controls challenge.

## Pygame Trajectory Viewer (`viz_realtime.py`)

Real-time 2D animated replay of a simulation rollout. Runs the full simulation first, then lets you scrub through the trajectory with playback controls.

### What it shows

- **Left pane** — Top-down trajectory view:
  - Target path (dotted gray) vs actual path (color-coded green→yellow→red by tracking error)
  - Car sprite with heading, glow effect, and breadcrumb trail
  - Trailing fade effect on recent path segments
- **Right panel** — Live dashboard:
  - Step progress bar with control-start marker
  - Velocity readout (m/s + km/h)
  - Center-fill gauge bars for steering and lateral acceleration (with red diamond target marker)
  - Mini sparkline chart of recent lataccel history
  - Running cost cards (LAT / JERK / TOTAL) that update as playback advances
  - WARMUP / CONTROL region badge

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play / pause |
| `←` `→` | Step backward / forward |
| `R` | Restart from beginning |
| `+` `-` | Speed up / slow down (0.5x–4x) |
| `ESC` | Quit |

### Usage

```bash
uv run viz_realtime.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller pid
```

Options:
- `--controller` — any controller in `controllers/` (default: `pid`)
- `--log_level` — `DEBUG`, `INFO`, or `WARNING` (default: `INFO`)

---

## Dash Dashboard (`viz_dashboard.py`)

Interactive web dashboard for controller tuning, analysis, and comparison. Runs on `http://localhost:8050`.

### Features

- **Single segment simulation** — pick a controller and data file, click Run, see 5 interactive Plotly charts:
  - Lateral acceleration (target vs actual)
  - Steering commands
  - Jerk
  - Velocity
  - Tracking error (with fill)
  - All charts have a control-start dashed line and cost-region shading
- **PID tuning** — P/I/D sliders with auto-update toggle. Adjust gains and see cost changes instantly without modifying any files.
- **Cost cards** — lataccel, jerk, and total cost displayed prominently
- **Batch mode** (collapsible) — run N segments, see cost distribution histograms
- **Comparison mode** (collapsible) — overlay two controllers on the same lataccel chart
- **Caching** — 50-entry LRU cache avoids re-running identical simulations

### Usage

```bash
uv run viz_dashboard.py
# open http://localhost:8050
```

Options:
- `--model_path` — path to ONNX model (default: `./models/tinyphysics.onnx`)
- `--data_dir` — path to data directory (default: `./data`)
- `--port` — server port (default: `8050`)
- `--debug` — enable Dash debug/hot-reload mode
- `--log_level` — `DEBUG`, `INFO`, or `WARNING` (default: `INFO`)

---

## Rerun Viewer (`viz_rerun.py`)

Interactive viewer built on [Rerun](https://rerun.io/) with native timeline scrubbing, multi-controller comparison, and 2D trajectory rendering. Saves a `.rrd` file and opens it in the Rerun viewer.

### What it shows

- **Left pane** — 2D trajectory view:
  - Target path (gray) and actual path (color-coded green→yellow→red by tracking error)
  - Animated position marker that moves with the timeline
- **Right pane** — 5 stacked time-series charts:
  - Lateral acceleration (target + actual overlaid)
  - Steering commands
  - Jerk
  - Velocity
  - Tracking error
- **Timeline scrubber** — drag to animate through simulation steps; play/pause controls built into Rerun

### Usage

```bash
# Single controller, single file
uv run viz_rerun.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller pid

# Compare two controllers side-by-side
uv run viz_rerun.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller pid zero

# Batch mode — first 5 segments
uv run viz_rerun.py --model_path ./models/tinyphysics.onnx --data_path ./data/ --controller pid --num_segs 5
```

Options:
- `--controller` — one or more controllers from `controllers/` (default: `pid`)
- `--num_segs` — number of segments to process in batch mode (default: `10`)

---

## Dependencies

All scripts use [inline script metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/) so `uv run` installs dependencies automatically. No changes to the project's `requirements.txt` are needed.

Additional packages used: `pygame`, `plotly`, `dash`, `rerun-sdk`.
