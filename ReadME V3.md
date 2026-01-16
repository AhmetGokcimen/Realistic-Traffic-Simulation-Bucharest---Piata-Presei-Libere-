# Bucharest Traffic Simulation 🇷🇴

An interactive traffic simulation based on real-world map data from **Piața Presei Libere, Bucharest**. Built with Python, Pygame, and OSMnx.

## Features 🌟
- **Real-Map Data:** Uses OpenStreetMap data via `osmnx` to simulate traffic on actual road networks.
- **Interactive Controls:** - **God Mode:** Right-click on traffic lights to toggle them manually.
  - **Traffic Density:** Use on-screen buttons to adjust car counts (50-250 cars).
- **Smart AI:** Cars follow the Intelligent Driver Model (IDM) for realistic acceleration and braking.
- **Visuals:** Heatmap overlays showing traffic density and randomized vehicle aesthetics.

## Installation 🛠️

1. Clone the repository:
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/REPO_ISMIN.git](https://github.com/KULLANICI_ADIN/REPO_ISMIN.git)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the simulation:
   ```bash
   python main.py
   ```

## Controls 🎮
- **Left Click + Drag:** Move the camera.
- **Mouse Wheel:** Zoom in/out.
- **Right Click (on Lights):** Change traffic light color.
- **Left Click (on Buttons):** Set total number of cars.
