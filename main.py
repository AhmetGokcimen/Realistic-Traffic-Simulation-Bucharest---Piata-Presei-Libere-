import sys
import time
import math
import random
import os
import traceback

# --- CRASH LOGGER ---
def log_crash(msg):
    with open("crash_log.txt", "w") as f:
        f.write(msg)

def exception_hook(exctype, value, tb):
    log_crash("".join(traceback.format_exception(exctype, value, tb)))
    sys.exit(1)

sys.excepthook = exception_hook

if sys.stdout:
    sys.stdout.reconfigure(line_buffering=True)

try:
    import osmnx as ox
    import networkx as nx
    from shapely.geometry import LineString
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
except:
    log_crash(f"Import Error:\n{traceback.format_exc()}")
    sys.exit(1)

ox.settings.use_cache = True
ox.settings.log_console = False

# --- CONFIGURATION PARAMETERS ---
# Initial size, but we will use dynamic size in the loop
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720 
METERS_TO_PIXELS = 12.0

# --- COLOR PALETTE ---
COLOR_BG = (16, 16, 16)
COLOR_ASPHALT = (32, 32, 32)
COLOR_ORANGE = (255, 165, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_CURB = (0, 0, 0)
COLOR_TEXT = (220, 220, 220)
COLOR_BTN_IDLE = (50, 50, 50)
COLOR_BTN_HOVER = (80, 80, 80)
COLOR_BTN_ACTIVE = (100, 200, 100)
COLOR_HEADLIGHT = (255, 255, 200, 40)
COLOR_PANEL_BG = (10, 10, 30, 200)
COLOR_PANEL_BORDER = (50, 50, 100)

# --- SIMULATION SETTINGS ---
SPEED_LEVELS = [0.5, 0.75, 1.0, 1.5, 2.0]

BRAND_COLORS = {
    "Mercedes": (200, 200, 200),
    "BMW": (80, 160, 255),
    "Volkswagen": (0, 120, 255),
    "Dacia": (170, 170, 170),
    "Renault": (255, 210, 0),
    "Audi": (220, 220, 255),
    "Toyota": (255, 80, 80),
    "Hyundai": (120, 255, 180),
}

BRANDS = list(BRAND_COLORS.keys())

# --- UI COMPONENT: BUTTON ---
class Button:
    def __init__(self, x, y, w, h, text, value):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.value = value
        self.is_hovered = False

    def draw(self, screen, font, current_val):
        color = COLOR_BTN_IDLE
        if self.value == current_val:
            color = COLOR_BTN_ACTIVE
        elif self.is_hovered:
            color = COLOR_BTN_HOVER

        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (200,200,200), self.rect, 2, border_radius=5)

        txt_surf = font.render(self.text, True, (255,255,255))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def check_hover(self, mx, my):
        self.is_hovered = self.rect.collidepoint(mx, my)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                return True
        return False

# --- SIMULATION AGENTS ---

class TrafficLight:
    def __init__(self, node_id, green_time=20.0, red_time=20.0):
        self.node_id = node_id
        self.state = 0 
        self.timer = random.uniform(0, green_time)
        self.green_time = green_time
        self.yellow_time = 2.0
        self.red_time = red_time
        self.sensor_active = False

    def update(self, dt):
        speed_factor = 1.0
        if self.state == 2 and self.sensor_active:
            speed_factor = 3.0

        self.timer -= dt * speed_factor
        self.sensor_active = False

        if self.timer <= 0:
            if self.state == 0:
                self.state = 1
                self.timer = self.yellow_time
            elif self.state == 1:
                self.state = 2
                self.timer = self.red_time
            elif self.state == 2:
                self.state = 0
                self.timer = self.green_time

    def get_color(self):
        if self.state == 0: return COLOR_GREEN
        if self.state == 1: return (255, 255, 0)
        return COLOR_RED

class Car:
    def __init__(self, car_id, G):
        self.id = car_id
        self.G = G
        self.length_m = random.uniform(3.8, 5.0)
        self.width_m = random.uniform(1.8, 2.3)
        self.brand = random.choice(BRANDS)
        self.color = BRAND_COLORS[self.brand]

        self.finished = False
        self.velocity = 0.0
        self.max_velocity = 8.0 

        self.a_max = 2.0   
        self.b_comf = 2.0  
        self.s0 = 1.5      
        self.T = 1.0       

        self.waiting_time = 0.0
        self.is_aggressive = False
        self.path = []

        self.turn_signal = 0
        self.blinker_timer = 0.0
        self.blinker_state = False

    def spawn_scatter(self):
        edges = list(self.G.edges(keys=True, data=True))
        if not edges: return False
        weights = [d['length'] for u,v,k,d in edges]

        for _ in range(10):
             u, v, k, data = random.choices(edges, weights=weights, k=1)[0]
             length = data['length']
             start_prog = random.uniform(0.0, max(0.0, length - 5.0))
             if length < 5.0: continue

             nodes = list(self.G.nodes())
             dest = random.choice(nodes)
             if dest == u or dest == v: continue

             try:
                 partial_path = nx.shortest_path(self.G, v, dest, weight='length')
                 full_path = [u] + partial_path
                 self.reset(full_path, start_prog)
                 return True
             except: pass
        return False

    def reset(self, path, start_progress=0.0):
        self.path = path
        self.path_index = 0
        self.progress = start_progress
        self.finished = False
        self.velocity = random.uniform(3.0, 8.0)
        self.waiting_time = 0.0
        self.is_aggressive = False

        u = self.path[0]
        v = self.path[1]
        self.current_edge_len = self.G.edges[u,v,0]['length']
        self._calculate_turn_signal()

    def _get_edge_len(self, u, v):
        return self.G.edges[u,v,0]['length']

    def _calculate_turn_signal(self):
        # --- BMW LOGIC IMPLEMENTATION ---
        # If it's a BMW, turn signals are physically disabled :)
        if self.brand == "BMW":
            self.turn_signal = 0
            return
        
        if self.path_index + 2 >= len(self.path):
            self.turn_signal = 0
            return

        u = self.path[self.path_index]
        v = self.path[self.path_index+1]
        w = self.path[self.path_index+2]

        n1 = self.G.nodes[u]
        n2 = self.G.nodes[v]
        n3 = self.G.nodes[w]

        v1x, v1y = n2['x'] - n1['x'], n2['y'] - n1['y']
        v2x, v2y = n3['x'] - n2['x'], n3['y'] - n2['y']

        ang1 = math.degrees(math.atan2(v1y, v1x))
        ang2 = math.degrees(math.atan2(v2y, v2x))

        diff = (ang2 - ang1 + 180) % 360 - 180

        if diff > 20: self.turn_signal = -1 
        elif diff < -20: self.turn_signal = 1 
        else: self.turn_signal = 0

    def update_physics(self, dt, traffic_sim):
        if self.finished or not self.path: return

        global_mult = getattr(traffic_sim, "speed_multiplier", 1.0)
        if global_mult <= 0: global_mult = 1.0

        brand_mult = 1.0
        if hasattr(traffic_sim, "brand_speed"):
            brand_mult = traffic_sim.brand_speed.get(self.brand, 1.0)

        speed_mult = global_mult * brand_mult
        v_cap = self.max_velocity * speed_mult

        self.blinker_timer += dt
        if self.blinker_timer > 0.4:
            self.blinker_timer = 0
            self.blinker_state = not self.blinker_state

        u = self.path[self.path_index]
        if self.path_index + 1 >= len(self.path):
            self.finished = True
            return
        v = self.path[self.path_index+1]

        edge_key = (u,v)
        cars_here = traffic_sim.cars_on_edge.get(edge_key, [])
        try: my_idx = cars_here.index(self)
        except: my_idx = -1

        leader_vel = v_cap
        gap = 1000.0 

        if self.velocity < 0.1: self.waiting_time += dt
        elif self.velocity > 1.0:
             self.waiting_time = 0.0
             self.is_aggressive = False

        if self.waiting_time > 5.0: self.is_aggressive = True

        dist_to_end = self.current_edge_len - self.progress
        is_green = False
        if v in traffic_sim.traffic_lights:
             if traffic_sim.traffic_lights[v].state == 0: is_green = True

        should_stop_light = False
        if my_idx > 0:
            leader = cars_here[my_idx - 1]
            gap = leader.progress - self.progress - leader.length_m
            leader_vel = leader.velocity
        else:
            if v in traffic_sim.traffic_lights:
                tl = traffic_sim.traffic_lights[v]
                if tl.state == 2: 
                    should_stop_light = True
                    if dist_to_end < 20.0:
                        tl.sensor_active = True 

                elif tl.state == 1: 
                    if (dist_to_end / max(self.velocity, 2.0)) > 2.0 or dist_to_end > 10.0:
                         should_stop_light = True

            if should_stop_light:
                if dist_to_end < gap:
                    gap = dist_to_end
                    leader_vel = 0.0

            elif not is_green and not self.is_aggressive and v not in traffic_sim.traffic_lights:
                if self.path_index + 2 < len(self.path):
                    v_next = self.path[self.path_index+2]
                    next_key = (v, v_next)
                    cars_next = traffic_sim.cars_on_edge.get(next_key, [])
                    if cars_next:
                        next_leader = cars_next[-1]
                        proj_gap = dist_to_end + next_leader.progress - next_leader.length_m
                        if proj_gap < gap:
                             gap = proj_gap
                             leader_vel = next_leader.velocity

        if gap < 0.01: gap = 0.01

        delta_v = self.velocity - leader_vel
        s_star = self.s0 + self.velocity * self.T + (self.velocity * delta_v) / (2*math.sqrt(self.a_max * self.b_comf))
        acc = self.a_max * (1 - (self.velocity / max(v_cap, 1e-6))**4 - (s_star / gap)**2)

        if is_green:
             if my_idx == 0 or gap > 20.0:
                  if self.velocity < 5.0 * speed_mult:
                       acc = self.a_max * 3.0
                  if self.velocity < 0.1 and gap > 5.0:
                       self.velocity = min(3.0 * speed_mult, v_cap)
                       self.progress += 0.1

        self.velocity += acc * dt
        if self.velocity < 0: self.velocity = 0
        if self.velocity > v_cap: self.velocity = v_cap

        if gap < 0.5 and self.velocity > 0:
             self.velocity = 0

        step_dist = self.velocity * dt
        if step_dist > gap: step_dist = max(0, gap - 0.1)

        self.progress += step_dist

        if self.progress >= self.current_edge_len:
             self.path_index += 1
             if self.path_index >= len(self.path)-1:
                  self.finished = True
             else:
                  self.progress = 0.0
                  self.current_edge_len = self._get_edge_len(self.path[self.path_index], self.path[self.path_index+1])
                  self._calculate_turn_signal()

    def get_state_visual(self):
        if not self.path or self.finished: return 0,0,0
        u = self.path[self.path_index]
        v = self.path[self.path_index+1]
        data = self.G.edges[u,v,0]
        geom = data.get('geometry')
        OFFSET_M = 1.5 

        if geom:
             frac = self.progress / self.current_edge_len
             frac = max(0, min(1, frac))
             pt = geom.interpolate(frac, normalized=True)
             pt_ex = geom.interpolate(min(frac+0.01, 1.0), normalized=True)
             
             dx, dy = pt_ex.x - pt.x, pt_ex.y - pt.y
             angle = 0 if (dx==0 and dy==0) else math.degrees(math.atan2(dy, dx))
             
             rad = math.radians(angle)
             ox = -math.sin(rad)*OFFSET_M
             oy = math.cos(rad)*OFFSET_M
             return pt.x + ox, pt.y + oy, angle
        else:
             n1, n2 = self.G.nodes[u], self.G.nodes[v]
             frac = self.progress / self.current_edge_len
             bx = n1['x'] + (n2['x'] - n1['x'])*frac
             by = n1['y'] + (n2['y'] - n1['y'])*frac
             dx, dy = n2['x']-n1['x'], n2['y']-n1['y']
             angle = math.degrees(math.atan2(dy, dx))
             rad = math.radians(angle)
             ox = -math.sin(rad)*OFFSET_M
             oy = math.cos(rad)*OFFSET_M
             return bx + ox, by + oy, angle

class TrafficSim:
    def __init__(self, G):
        self.G = G
        self.target_car_count = 250
        self.cars = []
        self.cars_on_edge = {}
        self.traffic_lights = {}
        self.is_night = False
        self.speed_multiplier = 1.0

        self.brand_speed = {b: 1.0 for b in BRANDS}
        self.brand_speed_idx = {b: SPEED_LEVELS.index(1.0) for b in BRANDS}

        self.visible_brand = None  

        self._init_tl()

        nodes = list(self.G.nodes(data=True))
        xs = [d['x'] for n, d in nodes]
        ys = [d['y'] for n, d in nodes]
        self.camera_x = sum(xs)/len(xs)
        self.camera_y = sum(ys)/len(ys)
        self.zoom = 12.0
        self.dragging = False
        self.ds_mouse = (0,0)
        self.ds_cam = (0,0)

    def _init_tl(self):
         for n in self.G.nodes():
              if self.G.degree[n] >= 3:
                   self.traffic_lights[n] = TrafficLight(n)

    def spawn_car(self):
        c = Car(len(self.cars), self.G)
        if c.spawn_scatter():
             self.cars.append(c)

    def step(self, dt):
        for tl in self.traffic_lights.values(): tl.update(dt)

        self.cars = [c for c in self.cars if not c.finished]
        while len(self.cars) < self.target_car_count:
            self.spawn_car()
        if len(self.cars) > self.target_car_count:
            self.cars = self.cars[:self.target_car_count]

        self.cars_on_edge = {}
        active = [c for c in self.cars if c.path and not c.finished]
        for c in active:
             k = (c.path[c.path_index], c.path[c.path_index+1])
             if k not in self.cars_on_edge: self.cars_on_edge[k] = []
             self.cars_on_edge[k].append(c)
        
        for k in self.cars_on_edge:
             self.cars_on_edge[k].sort(key=lambda x: x.progress, reverse=True)
             
        for c in active: c.update_physics(dt, self)

    def get_statistics(self):
        total_cars = len(self.cars)
        if total_cars == 0: return 0, 0, 0

        stopped_cars = 0
        total_velocity = 0
        for c in self.cars:
            total_velocity += c.velocity
            if c.velocity < 1.0:
                stopped_cars += 1

        avg_speed = (total_velocity / total_cars) * 3.6 
        congestion = stopped_cars / total_cars

        return total_cars, avg_speed, congestion

    def toggle_light_at_cursor(self, mx, my, scr_w, scr_h):
        # Passed Screen width/height for accurate Raycasting
        wx, wy = screen_to_world(mx, my, self.camera_x, self.camera_y, self.zoom, scr_w, scr_h)
        closest_node = None
        min_dist = 30.0
        for node_id, tl in self.traffic_lights.items():
            nx = self.G.nodes[node_id]['x']
            ny = self.G.nodes[node_id]['y']
            dist = math.hypot(nx - wx, ny - wy)
            if dist < min_dist:
                min_dist = dist
                closest_node = tl

        if closest_node:
            if closest_node.state == 0:
                closest_node.state = 2
                closest_node.timer = closest_node.red_time
            else:
                closest_node.state = 0
                closest_node.timer = closest_node.green_time
            return True
        return False

# --- UTILITY FUNCTIONS ---
def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

def world_to_screen(wx, wy, cx, cy, zoom, scr_w, scr_h):
    """ Projects World Coordinates using dynamic Screen Pixels """
    sx = (scr_w // 2) + (wx - cx) * zoom
    sy = (scr_h // 2) + (cy - wy) * zoom
    return int(sx), int(sy)

def screen_to_world(sx, sy, cx, cy, zoom, scr_w, scr_h):
    """ Projects Screen Pixels using dynamic Screen Dimensions """
    wx = cx + (sx - scr_w/2) / zoom
    wy = cy - (sy - scr_h/2) / zoom
    return wx, wy

def rotate_point(px, py, angle):
    rad = math.radians(angle)
    c = math.cos(rad)
    s = math.sin(rad)
    return px * c - py * s, px * s + py * c

# --- DASHBOARD RENDERING ---
def draw_dashboard(screen, sim, font):
    """ Draws the Analytics Panel anchored to Top Right """
    scr_w, scr_h = screen.get_size()
    
    w, h = 240, 150
    # Anchor to Top Right
    x, y = scr_w - w - 10, 10

    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill(COLOR_PANEL_BG)
    pygame.draw.rect(s, COLOR_PANEL_BORDER, (0, 0, w, h), 2)

    total, avg_speed, congestion = sim.get_statistics()

    lines = [
        f"ANALYTICS PANEL",
        f"-----------------------",
        f"Active Cars:  {total} / {sim.target_car_count}",
        f"Avg Speed:    {avg_speed:.1f} km/h",
        f"Congestion:   {int(congestion*100)}%"
    ]

    for i, line in enumerate(lines):
        col = (200, 255, 255) if i == 0 else (220, 220, 220)
        txt = font.render(line, True, col)
        s.blit(txt, (15, 15 + i * 22))

    bar_w = 200
    bar_h = 10
    bar_x = 15
    bar_y = 125

    pygame.draw.rect(s, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))

    fill_w = int(bar_w * congestion)
    fill_col = lerp_color(COLOR_GREEN, COLOR_RED, congestion)
    if fill_w > 0:
        pygame.draw.rect(s, fill_col, (bar_x, bar_y, fill_w, bar_h))

    pygame.draw.rect(s, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h), 1)

    screen.blit(s, (x, y))

# --- MAIN APPLICATION LOOP ---
def main():
    graph_file = "traffic_graph.graphml"
    
    if getattr(sys, 'frozen', False) and os.path.exists(os.path.join(sys._MEIPASS, graph_file)):
        G = ox.load_graphml(os.path.join(sys._MEIPASS, graph_file))
    elif os.path.exists(graph_file):
        G = ox.load_graphml(graph_file)
    else:
        G = ox.graph_from_point((44.478121, 26.072711), dist=450, network_type='drive')
        ox.save_graphml(G, graph_file)

    G_proj = ox.project_graph(G)
    for u, v, k, d in G_proj.edges(keys=True, data=True):
         if 'geometry' not in d:
              d['geometry'] = LineString([(G_proj.nodes[u]['x'], G_proj.nodes[u]['y']),
                                          (G_proj.nodes[v]['x'], G_proj.nodes[v]['y'])])

    sim = TrafficSim(G_proj)

    pygame.init()
    # Initial setup
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("Traffic Sim V4: Analytics Dashboard")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('consolas', 14, bold=True)

    # Initialize Buttons (Coordinates will be updated in loop)
    buttons = []
    vals = [50, 100, 150, 200, 250]
    for v in vals:
        buttons.append(Button(0, 0, 50, 30, str(v), v))

    brand_buttons = []
    bw, bh = 150, 28
    gap_y = 6
    start_x, start_y = 10, 10
    for i, brand in enumerate(BRANDS):
        y = start_y + i * (bh + gap_y)
        brand_buttons.append(Button(start_x, y, bw, bh, brand, brand))

    speed_buttons = []
    speed_vals = [0.5, 0.75, 1.0, 1.5, 2.0]
    btn_w, btn_h = 60, 30
    btn_gap = 10
    total_speed_w = len(speed_vals) * btn_w + (len(speed_vals) - 1) * btn_gap

    for sv in speed_vals:
        speed_buttons.append(Button(0, 0, btn_w, btn_h, f"x{sv}", sv))

    running = True
    while running:
        dt = 0.1 
        mx, my = pygame.mouse.get_pos()
        current_w, current_h = screen.get_size() # Get current Dynamic Resolution

        # --- DYNAMIC UI LAYOUT UPDATE ---
        # 1. Car Count Buttons -> Bottom Left
        bx, by = 10, current_h - 40
        for b in buttons:
            b.rect.topleft = (bx, by)
            bx += 60

        # 2. Speed Buttons -> Bottom Right
        sbx = current_w - total_speed_w - 10
        sby = current_h - 44
        for b in speed_buttons:
            b.rect.topleft = (sbx, sby)
            sbx += btn_w + btn_gap

        # Brand buttons are already fixed Top Left (start_x, start_y), so they are fine.
        
        # Handle Input
        for b in buttons: b.check_hover(mx, my)
        for b in speed_buttons: b.check_hover(mx, my)
        for b in brand_buttons: b.check_hover(mx, my)

        for e in pygame.event.get():
             if e.type == pygame.QUIT:
                 running = False

             elif e.type == pygame.KEYDOWN:
                 if e.key == pygame.K_n:
                     sim.is_night = not sim.is_night

             elif e.type == pygame.MOUSEWHEEL:
                  wx_b, wy_b = screen_to_world(mx, my, sim.camera_x, sim.camera_y, sim.zoom, current_w, current_h)
                  if e.y > 0: sim.zoom = min(sim.zoom * 1.1, 50.0)
                  elif e.y < 0: sim.zoom = max(sim.zoom / 1.1, 0.5)
                  wx_n, wy_n = screen_to_world(mx, my, sim.camera_x, sim.camera_y, sim.zoom, current_w, current_h)
                  sim.camera_x -= (wx_n - wx_b)
                  sim.camera_y -= (wy_n - wy_b)

             elif e.type == pygame.MOUSEBUTTONDOWN:
                  if e.button == 1:
                      clicked_btn = False
                      for b in buttons:
                          if b.is_clicked(e):
                              sim.target_car_count = b.value
                              clicked_btn = True
                              break
                      if not clicked_btn:
                          for b in speed_buttons:
                              if b.is_clicked(e):
                                  sim.speed_multiplier = b.value
                                  clicked_btn = True
                                  break
                      if not clicked_btn:
                          for b in brand_buttons:
                              if b.is_clicked(e):
                                  br = b.value
                                  sim.visible_brand = None if sim.visible_brand == br else br
                                  clicked_btn = True
                                  break
                      if not clicked_btn:
                          sim.dragging = True
                          sim.ds_mouse = (mx, my)
                          sim.ds_cam = (sim.camera_x, sim.camera_y)

                  elif e.button == 3:
                      sim.toggle_light_at_cursor(mx, my, current_w, current_h)

             elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                  sim.dragging = False

        if sim.dragging:
             dx = mx - sim.ds_mouse[0]
             dy = my - sim.ds_mouse[1]
             sim.camera_x = sim.ds_cam[0] - dx / sim.zoom
             sim.camera_y = sim.ds_cam[1] + dy / sim.zoom

        sim.step(dt)

        # --- RENDERING PIPELINE ---
        screen.fill(COLOR_BG)
        ROAD_W = int(3.5 * sim.zoom)
        CURB_W = ROAD_W + 4

        # Pass current_w and current_h to world_to_screen functions
        visible_edges = []
        for u, v, k, d in G_proj.edges(keys=True, data=True):
             g = d['geometry']
             if g.geom_type == 'LineString':
                  pts = [world_to_screen(x,y, sim.camera_x, sim.camera_y, sim.zoom, current_w, current_h) for x,y in g.coords]
                  # Frustum culling using current screen dims
                  if any(-100 < p[0] < current_w+100 and -100 < p[1] < current_h+100 for p in pts):
                       visible_edges.append((u, v, d, pts))
                       if len(pts) > 1:
                            pygame.draw.lines(
                                screen,
                                (80, 80, 80) if sim.is_night else COLOR_CURB,
                                False,
                                pts,
                                CURB_W + (2 if sim.is_night else 0)
                            )
                            pygame.draw.lines(screen, COLOR_ASPHALT, False, pts, ROAD_W)

        heatmap_surf = pygame.Surface((current_w, current_h), pygame.SRCALPHA)
        for u, v, d, pts in visible_edges:
             if len(pts) > 1:
                  cars = sim.cars_on_edge.get((u,v), [])
                  cnt = len(cars)
                  capacity = d['length'] / 7.0
                  density = 0
                  if capacity > 0: density = cnt / capacity

                  if density > 0.1:
                       col = (0,0,0,0)
                       if density >= 0.8:
                            col = (*COLOR_RED, 100)
                       elif density >= 0.2:
                            t = (density - 0.2) / 0.6
                            rgb = lerp_color(COLOR_ORANGE, COLOR_RED, t) if density > 0.5 else COLOR_ORANGE
                            col = (*rgb, 100)
                       else:
                            col = (*COLOR_ORANGE, 50)
                       pygame.draw.lines(heatmap_surf, col, False, pts, ROAD_W)
        screen.blit(heatmap_surf, (0,0))

        for u, v, d, pts in visible_edges:
             if len(pts) > 1:
                  for i in range(len(pts)-1):
                       p1, p2 = pts[i], pts[i+1]
                       dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                       if dist > 10:
                            steps = int(dist / 20)
                            for s in range(1, steps):
                                 t = s / steps
                                 mx_p = int(p1[0] + (p2[0]-p1[0])*t)
                                 my_p = int(p1[1] + (p2[1]-p1[1])*t)
                                 pygame.draw.circle(screen, (80, 80, 80), (mx_p, my_p), 1)

        light_surf = pygame.Surface((current_w, current_h), pygame.SRCALPHA)

        for c in sim.cars:
             if sim.visible_brand is not None and getattr(c, "brand", None) != sim.visible_brand:
                 continue
             if c.finished: continue
             wx, wy, ang = c.get_state_visual()
             sx, sy = world_to_screen(wx, wy, sim.camera_x, sim.camera_y, sim.zoom, current_w, current_h)

             if -100 < sx < current_w+100 and -100 < sy < current_h+100:
                  L = int(c.length_m * sim.zoom)
                  W = int(c.width_m * sim.zoom)

                  if sim.is_night:
                      cone_len = L * 2.5
                      cone_w = W * 1.8
                      fl_x, fl_y = L/2, -W/4
                      fr_x, fr_y = L/2, W/4
                      p1 = rotate_point(fl_x, fl_y, -ang)
                      p2 = rotate_point(fr_x, fr_y, -ang)
                      p3 = rotate_point(fl_x + cone_len, fl_y - cone_w/2, -ang)
                      p4 = rotate_point(fr_x + cone_len, fr_y + cone_w/2, -ang)
                      poly = [
                          (sx + p1[0], sy + p1[1]),
                          (sx + p3[0], sy + p3[1]),
                          (sx + p4[0], sy + p4[1]),
                          (sx + p2[0], sy + p2[1])
                      ]
                      pygame.draw.polygon(light_surf, COLOR_HEADLIGHT, poly)

                  s = pygame.Surface((L, W), pygame.SRCALPHA)
                  pygame.draw.rect(s, (0,0,0), (0,0,L,W), border_radius=2)

                  base_col = c.color
                  if c.is_aggressive: base_col = (255, 50, 50)
                  pygame.draw.rect(s, base_col, (1,1,L-2,W-2), border_radius=2)

                  if c.velocity < 0.1: 
                       pygame.draw.rect(s, (255,0,0), (1,1,3,W//3))
                       pygame.draw.rect(s, (255,0,0), (1,W-1-W//3,3,W//3))

                  # Turn signal rendering
                  if c.blinker_state and c.turn_signal != 0:
                      blink_col = (255, 255, 0)
                      if c.turn_signal == 1:
                           pygame.draw.rect(s, blink_col, (L-5, W-5, 4, 4))
                           pygame.draw.rect(s, blink_col, (1, W-5, 4, 4))
                      elif c.turn_signal == -1:
                           pygame.draw.rect(s, blink_col, (L-5, 1, 4, 4))
                           pygame.draw.rect(s, blink_col, (1, 1, 4, 4))

                  rs = pygame.transform.rotate(s, ang)
                  screen.blit(rs, rs.get_rect(center=(sx, sy)))

        if sim.is_night:
            night_surf = pygame.Surface((current_w, current_h), pygame.SRCALPHA)
            night_surf.fill((0, 0, 20, 170))
            screen.blit(night_surf, (0,0))
            screen.blit(light_surf, (0,0))

        for n, d in G_proj.nodes(data=True):
             sx, sy = world_to_screen(d['x'], d['y'], sim.camera_x, sim.camera_y, sim.zoom, current_w, current_h)
             if -50 < sx < current_w+50 and -50 < sy < current_h+50:
                  rad = int(ROAD_W * 0.8)
                  if n in sim.traffic_lights:
                       l = sim.traffic_lights[n]
                       if sim.is_night:
                           pygame.draw.circle(screen, l.get_color(), (sx, sy), rad * 1.8, width=0)
                       pygame.draw.circle(screen, (0,0,0), (sx, sy), rad+2)
                       pygame.draw.circle(screen, l.get_color(), (sx, sy), rad)
                  else:
                       pygame.draw.circle(screen, COLOR_CURB, (sx, sy), 2)

        # UI OVERLAY - Use current_w and current_h
        draw_dashboard(screen, sim, font)

        info_txt = font.render("Left Click: Set Cars | Right Click: Light | N: Night Mode", True, (150, 150, 150))
        screen.blit(info_txt, (10, current_h - 70))

        for b in buttons:
            b.draw(screen, font, sim.target_car_count)

        for b in brand_buttons:
            br = b.value
            b.text = f"{br}"
            b.draw(screen, font, sim.visible_brand)

        sx = current_w - total_speed_w - 10
        sy = current_h - 70

        speed_txt = font.render("Speed:", True, (150, 150, 150))
        screen.blit(speed_txt, (sx, sy))

        for b in speed_buttons:
            b.draw(screen, font, sim.speed_multiplier)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()