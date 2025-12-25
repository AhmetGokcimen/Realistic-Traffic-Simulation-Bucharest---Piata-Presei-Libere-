# 🚦 Realistic Traffic Simulation (Bucharest)

A high-fidelity traffic simulation visualizing traffic flow in the Herastrau
area of Bucharest. This project uses OpenStreetMap data for realistic road
geometry and implements custom physics for preventing gridlocks and ensuring
smooth visual flow.

## ✨ Features

- **Realistic Physics**: Intelligent Driver Model (IDM) with anti-stacking
  logic.
- **Smart Pathfinding**: "Noisy" GPS logic ensures traffic spreads across all
  available roads.
- **Visual Polish**:
  - Strict "Theatre Stage" masking hides floating cars outside the map.
  - Cars follow the exact curvature of the road (Offset Synchronization).
- **Interactive Camera**: Zoom (Scroll) and Pan (Drag) to explore the map.
- **Data**: Real-world OSM data via `osmnx`.

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/traffic_sim.git
   cd traffic_sim
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Simulation

Run the main script:

```bash
python main.py
```

## 🎮 Controls

- **Left Click + Drag**: Pan the camera.
- **Scroll Wheel**: Zoom in/out.
- **Right Click**: Start/Stop GIF recording.

## 🛠️ Building the Executable

To create a standalone `.exe` file for sharing:

```bash
pyinstaller --onefile --copy-metadata osmnx --hidden-import=pygame --hidden-import=PIL main.py
```

The output file `main.exe` will be in the `dist/` folder.

## 📄 License

MIT
