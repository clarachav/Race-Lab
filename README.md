```markdown
# RaceLab — Motorsport Vehicle Dynamics & Lap Time Simulation

A Python-based vehicle dynamics, tyre modelling, and lap-time simulation engine built for motorsport engineering analysis and real-time race telemetry visualization.

---

## Overview

**RaceLab** models longitudinal and lateral vehicle performance around closed circuits (including **Circuit de Spa-Francorchamps** loaded from GeoJSON coordinates and parametric test tracks). It combines quasi-steady-state physics solvers with interactive multi-car race animations, real-time gap tracking, pit-exit position estimation, and live timing scoreboards.

### Key Capabilities

* **Physics & Vehicle Dynamics (`src/racelab/physics`, `src/racelab/models`)**
  * Aerodynamic downforce and drag modelling (`aerodynamics.py`)
  * Tyre grip sensitivity and load-dependent behaviour (`tires.py`)
  * Longitudinal and lateral load transfer calculations (`load_transfer.py`, `dynamics.py`)
  * Parametric vehicle setup and track curvature discretization (`vehicle.py`, `track.py`)
* **Lap Time & Performance Solvers (`src/racelab/simulation`)**
  * Straight-line acceleration and braking profiles (`straight_line.py`)
  * Cornering speed limits from lateral grip and track curvature (`cornering.py`)
  * Full flying-lap forward/backward velocity integration (`lap_time.py`, `lap_demo.py`)
* **Interactive Telemetry & Race Visualization (`lap_animation.py`)**
  * **Multi-Car 2025 Grid Simulation:** 20-car race replay around Spa-Francorchamps with live speed heatmaps and car trails.
  * **Interactive Timeline Controls:** Scrub through any point of the lap with a time slider, `-5s` / `+5s` jump buttons, keyboard shortcuts (`Left`/`Right`/`Space`), and `Play`/`Pause`.
  * **Real-Time Gap Map:** Horizontal time-delta axis displaying live gaps behind the leader.
  * **Pit Exit Position Estimation:** Live prediction of rejoin position and time delta to nearest traffic based on pit entry/exit markers and adjustable pit-lane time loss (`Pit -1s` / `Pit +1s`).
  * **Dedicated Live Scoreboards Window:** Dual-panel timing screen displaying live standings, intervals, speed traps, sector splits (`S1`, `S2`, `S3`), and estimated lap times.

---

## Repository Structure

```text
Race-Lab/
├── data/
│   └── raw/                  # Track geometry files (e.g., spa.geojson)
├── outputs/
│   └── figures/              # Generated telemetry plots and simulation exports
├── src/
│   └── racelab/
│       ├── analysis/         # Circuit data fetching and preprocessing
│       ├── models/           # Vehicle, tyre, aerodynamics, and track models
│       ├── physics/          # Longitudinal/lateral dynamics and load transfer
│       ├── simulation/       # Flying lap solver, cornering, straight-line & animation
│       └── constants.py      # Physical constants
├── tests/                    # Pytest unit and integration test suite
├── pyproject.toml            # Package configuration and dependencies
└── README.md

```

---

## Installation

1. **Clone the repository**
```bash
git clone [https://github.com/clarachav/Race-Lab.git](https://github.com/clarachav/Race-Lab.git)
cd Race-Lab

```


2. **Create and activate a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

```


3. **Install the package and dependencies**
```bash
pip install -e .

```



---

## Usage

### Run the Interactive Lap & Scoreboard Simulation

```bash
python -m racelab.simulation.lap_animation

```

* **Controls:**
* Drag the **Time `[s]**` slider or use **`-5s` / `+5s**` (or `Left` / `Right` arrow keys) to scrub the simulation.
* Press **`Space`** or click **`Pause` / `Play**` to toggle playback.
* Click **`Pit: <DRIVER>`** to cycle the target car for pit-exit rejoin estimation, and **`Pit -1s` / `Pit +1s**` to adjust pit-loss time.



### Run the Test Suite

```bash
pytest

```

---

## Engineering Roadmap

* [ ] Multi-lap tyre thermal and wear degradation model
* [ ] Fuel burn mass-reduction effects over full race stints
* [ ] Slipstream / DRS drag-reduction zones on straights
* [ ] Telemetry comparison against real-world FastF1 / open timing data

---

## Author

**Clara Chav** — Aerospace & Mechanical Engineering Portfolio

GitHub: [@clarachav](https://www.google.com/search?q=https://github.com/clarachav)

```

```
