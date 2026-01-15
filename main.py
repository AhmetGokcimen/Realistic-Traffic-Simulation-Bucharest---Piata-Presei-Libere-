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

# --- CONFIG ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
METERS_TO_PIXELS = 12.0

# --- COLORS ---
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

# --- UI BUTTON CLASS ---
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

# --- SIMULATION CLASSES ---

class TrafficLight:
    def __init__(self, node_id, green_time=20.0, red_time=20.0):
        self.node_id = node_id
        self.state = 0 # 0=Green, 1=Yellow, 2=Red
        self.timer = random.uniform(0, green_time)
        self.green_time = green_time
        self.yellow_time = 2.0
        self.red_time = red_time

    def update(self, dt):
        self.timer -= dt
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
        # Visual Customization
        self.length_m = random.uniform(3.8, 5.0) # Different lengths
        self.width_m = random.uniform(1.8, 2.3)
        self.color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        
        self.finished = False
        
        self.velocity = 0.0
        self.max_velocity = 8.0 # ~30km/h
        
        # IDM Parameters
        self.a_max = 2.0
        self.b_comf = 2.0
        self.s0 = 1.5 
        self.T = 1.0
        
        self.waiting_time = 0.0
        self.is_aggressive = False
        self.path = []

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

    def _get_edge_len(self, u, v):
        return self.G.edges[u,v,0]['length']

    def update_physics(self, dt, traffic_sim):
        if self.finished or not self.path: return
        
        u = self.path[self.path_index]
        if self.path_index + 1 >= len(self.path):
            self.finished = True
            return
        v = self.path[self.path_index+1]
        
        edge_key = (u,v)
        cars_here = traffic_sim.cars_on_edge.get(edge_key, [])
        try: my_idx = cars_here.index(self)
        except: my_idx = -1
        
        leader_vel = self.max_velocity
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

        if my_idx > 0:
            leader = cars_here[my_idx - 1]
            gap = leader.progress - self.progress - leader.length_m
            leader_vel = leader.velocity
        else:
            should_stop_light = False
            if v in traffic_sim.traffic_lights:
                tl = traffic_sim.traffic_lights[v]
                if tl.state == 2: # Red
                    should_stop_light = True
                elif tl.state == 1: # Yellow
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
        acc = self.a_max * (1 - (self.velocity / self.max_velocity)**4 - (s_star / gap)**2)
        
        if is_green:
             if my_idx == 0 or gap > 20.0:
                  if self.velocity < 5.0: 
                       acc = self.a_max * 3.0 
                  if self.velocity < 0.1 and gap > 5.0:
                       self.velocity = 3.0 
                       self.progress += 0.1
        
        self.velocity += acc * dt
        if self.velocity < 0: self.velocity = 0
        if self.velocity > self.max_velocity: self.velocity = self.max_velocity
        
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
        self.target_car_count = 250 # Default
        self.cars = []
        self.cars_on_edge = {}
        self.traffic_lights = {}
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
        
        # Cleanup and Target Count Control
        self.cars = [c for c in self.cars if not c.finished]
        
        # Spawn if below target
        while len(self.cars) < self.target_car_count: 
            self.spawn_car()
            
        # Remove if above target (removing from end is more performant)
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

    def toggle_light_at_cursor(self, mx, my):
        # Convert mouse coords to world coords
        wx, wy = screen_to_world(mx, my, self.camera_x, self.camera_y, self.zoom)
        
        closest_node = None
        min_dist = 30.0 # Click range (meters)
        
        for node_id, tl in self.traffic_lights.items():
            nx = self.G.nodes[node_id]['x']
            ny = self.G.nodes[node_id]['y']
            dist = math.hypot(nx - wx, ny - wy)
            
            if dist < min_dist:
                min_dist = dist
                closest_node = tl

        if closest_node:
            # Toggle state: Green to Red, others to Green
            if closest_node.state == 0:
                closest_node.state = 2
                closest_node.timer = closest_node.red_time
            else:
                closest_node.state = 0
                closest_node.timer = closest_node.green_time
            return True
        return False

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

def world_to_screen(wx, wy, cx, cy, zoom):
    sx = (SCREEN_WIDTH // 2) + (wx - cx) * zoom
    sy = (SCREEN_HEIGHT // 2) + (cy - wy) * zoom
    return int(sx), int(sy)

def screen_to_world(sx, sy, cx, cy, zoom):
    wx = cx + (sx - SCREEN_WIDTH/2) / zoom
    wy = cy - (sy - SCREEN_HEIGHT/2) / zoom
    return wx, wy

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
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Traffic Sim: God Mode & Controls")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('arial', 14, bold=True)
    
    # --- CREATE BUTTONS ---
    buttons = []
    vals = [50, 100, 150, 200, 250]
    bx, by = 10, SCREEN_HEIGHT - 40
    for v in vals:
        buttons.append(Button(bx, by, 50, 30, str(v), v))
        bx += 60 # Add spacing

    running = True
    while running:
        dt = 0.1
        mx, my = pygame.mouse.get_pos()
        
        # Button Hover Check
        for b in buttons: b.check_hover(mx, my)

        for e in pygame.event.get():
             if e.type == pygame.QUIT: running = False
             elif e.type == pygame.MOUSEWHEEL:
                  wx_b, wy_b = screen_to_world(mx, my, sim.camera_x, sim.camera_y, sim.zoom)
                  if e.y > 0: sim.zoom = min(sim.zoom * 1.1, 50.0)
                  elif e.y < 0: sim.zoom = max(sim.zoom / 1.1, 0.5)
                  wx_n, wy_n = screen_to_world(mx, my, sim.camera_x, sim.camera_y, sim.zoom)
                  sim.camera_x -= (wx_n - wx_b)
                  sim.camera_y -= (wy_n - wy_b)
             
             elif e.type == pygame.MOUSEBUTTONDOWN:
                  # Left Click (Buttons or Dragging)
                  if e.button == 1:
                      # Check if buttons are clicked first
                      clicked_btn = False
                      for b in buttons:
                          if b.is_clicked(e):
                              sim.target_car_count = b.value
                              clicked_btn = True
                              break
                      
                      # If no button clicked, drag camera
                      if not clicked_btn:
                          sim.dragging = True
                          sim.ds_mouse = (mx, my)
                          sim.ds_cam = (sim.camera_x, sim.camera_y)
                  
                  # Right Click (God Mode - Toggle Light)
                  elif e.button == 3:
                      sim.toggle_light_at_cursor(mx, my)

             elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                  sim.dragging = False
        
        if sim.dragging:
             dx = mx - sim.ds_mouse[0]
             dy = my - sim.ds_mouse[1]
             sim.camera_x = sim.ds_cam[0] - dx / sim.zoom
             sim.camera_y = sim.ds_cam[1] + dy / sim.zoom
             
        sim.step(dt)
        
        screen.fill(COLOR_BG)
        ROAD_W = int(3.5 * sim.zoom)
        CURB_W = ROAD_W + 4
        
        # --- LAYER 1: ROAD ---
        visible_edges = []
        for u, v, k, d in G_proj.edges(keys=True, data=True):
             g = d['geometry']
             if g.geom_type == 'LineString':
                  pts = [world_to_screen(x,y, sim.camera_x, sim.camera_y, sim.zoom) for x,y in g.coords]
                  if any(-100 < p[0] < SCREEN_WIDTH+100 and -100 < p[1] < SCREEN_HEIGHT+100 for p in pts):
                       visible_edges.append((u, v, d, pts))
                       if len(pts) > 1:
                            pygame.draw.lines(screen, COLOR_CURB, False, pts, CURB_W)
                            pygame.draw.lines(screen, COLOR_ASPHALT, False, pts, ROAD_W)
        
        # --- LAYER 2: HEATMAP ---
        heatmap_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
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
        
        # --- LAYER 3: MARKINGS ---
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

        # --- LAYER 4: OBJECTS ---
        for n, d in G_proj.nodes(data=True):
             sx, sy = world_to_screen(d['x'], d['y'], sim.camera_x, sim.camera_y, sim.zoom)
             if -50 < sx < SCREEN_WIDTH+50 and -50 < sy < SCREEN_HEIGHT+50:
                  rad = int(ROAD_W * 0.8)
                  if n in sim.traffic_lights:
                       l = sim.traffic_lights[n]
                       pygame.draw.circle(screen, (0,0,0), (sx, sy), rad+2)
                       pygame.draw.circle(screen, l.get_color(), (sx, sy), rad)
                  else:
                       pygame.draw.circle(screen, COLOR_CURB, (sx, sy), 2)

        for c in sim.cars:
             if c.finished: continue
             wx, wy, ang = c.get_state_visual()
             sx, sy = world_to_screen(wx, wy, sim.camera_x, sim.camera_y, sim.zoom)
             if -100 < sx < SCREEN_WIDTH+100 and -100 < sy < SCREEN_HEIGHT+100:
                  L = int(c.length_m * sim.zoom)
                  W = int(c.width_m * sim.zoom)
                  s = pygame.Surface((L, W), pygame.SRCALPHA)
                  pygame.draw.rect(s, (0,0,0), (0,0,L,W), border_radius=2)
                  
                  # Custom Color Usage
                  base_col = c.color
                  if c.is_aggressive: base_col = (255, 50, 50)
                  pygame.draw.rect(s, base_col, (1,1,L-2,W-2), border_radius=2)
                  
                  if c.velocity < 0.1:
                       pygame.draw.rect(s, (255,0,0), (1,1,3,W//3))
                       pygame.draw.rect(s, (255,0,0), (1,W-1-W//3,3,W//3))
                  
                  rs = pygame.transform.rotate(s, ang)
                  screen.blit(rs, rs.get_rect(center=(sx, sy)))

        # --- UI DRAWING ---
        txt = font.render(f"Zoom: {sim.zoom:.1f} | Cars: {len(sim.cars)} / {sim.target_car_count}", True, COLOR_TEXT)
        screen.blit(txt, (10, 10))
        
        info_txt = font.render("Right Click Lights to Toggle | Left Click Buttons to Set Cars", True, (150, 150, 150))
        screen.blit(info_txt, (10, 30))

        for b in buttons:
            b.draw(screen, font, sim.target_car_count)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()