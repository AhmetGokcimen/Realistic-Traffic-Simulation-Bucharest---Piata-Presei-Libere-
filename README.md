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

  ## 🌟 Key Features (V3)

### 🧠 Smart Traffic Systems
- **Real-World Map Data:** Fetches live road networks using `osmnx` and `networkx`.
- **Intelligent Driver Model (IDM):** Cars follow realistic acceleration, braking, and gap-keeping physics.
- **Adaptive Traffic Lights:** Lights automatically speed up the green cycle if sensors detect waiting cars.

### 🎮 Interactive Controls (God Mode)
- **God Mode:** Right-click on any traffic light to manually force it Green/Red.
- **Traffic Density Control:** Use on-screen buttons to adjust the number of active cars (50-250) in real-time.

### 📊 Analytics & Visuals
- **Live Dashboard:** Real-time monitoring of **Average Speed**, **Congestion Levels**, and **Active Vehicle Count**.
- **Day/Night Cycle:** Press `N` to toggle Night Mode, enabling dynamic headlights and street glow.
- **Turn Signals:** Vehicles automatically signal before making turns at intersections.


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

## Controls 🎮
- **Left Click + Drag:** Move the camera.
- **Mouse Wheel:** Zoom in/out.
- **Right Click (on Lights):** Change traffic light color.
- **Left Click (on Buttons):** Set total number of cars.
- **Press `N` to toggle Night Mode

## 🛠️ Building the Executable

To create a standalone `.exe` file for sharing:

```bash
pyinstaller --onefile --copy-metadata osmnx --hidden-import=pygame --hidden-import=PIL main.py
```

The output file `main.exe` will be in the `dist/` folder.

## 📄 License

MIT
