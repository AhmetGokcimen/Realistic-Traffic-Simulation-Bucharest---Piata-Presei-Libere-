import sys
import time
import math
import random
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print("Starting application...", flush=True)

try:
    print("Importing osmnx...", flush=True)
    import osmnx as ox
    print("Importing networkx...", flush=True)
    import networkx as nx
    print("Importing shapely...", flush=True)
    from shapely.geometry import LineString, MultiLineString
    print("Importing pygame...", flush=True)
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    import PIL.Image
    print("Imports complete.", flush=True)
except Exception as e:
    print(f"Error during imports: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

# Configure osmnx
ox.settings.use_cache = True
ox.settings.log_console = False

class TrafficLight:
    def __init__(self, node_id, cycle_time=30.0):
        self.node_id = node_id
        self.is_green = True
        self.cycle_time = cycle_time
        self.timer = random.uniform(0, cycle_time)

    def update(self, dt):
        self.timer -= dt
        if self.timer <= 0:
            self.is_green = not self.is_green
            self.timer = self.cycle_time

class Car:
    def __init__(self, car_id, origin_node, destination_node, G):
        self.id = car_id
        self.G = G
        self.length = 12.0 
        self.finished = False
        
        # Physics State
        self.velocity = 0.0
        self.max_velocity = 10.0 
        self.acceleration = 0.0
        
        # IDM Parameters
        self.a_max = 1.5
        self.b_comf = 2.0
        self.T = 1.0
        self.s0 = 2.0
        self.delta = 4.0

        # Init
        self.reset(origin_node, destination_node)

    def reset(self, origin_node, destination_node, path=None):
        self.origin = origin_node
        self.destination = destination_node
        self.finished = False
        self.velocity = 5.5 # Start with some speed (20km/h) to look like entering
        self.acceleration = 0.0
        
        try:
            if path:
                self.path = path
            else:
                self.path = nx.shortest_path(self.G, origin_node, destination_node, weight='length')
                
            self.path_index = 0 
            self.progress = 0.0
            self.current_edge_len = self._get_edge_len(self.path[0], self.path[1])
        except (nx.NetworkXNoPath, IndexError, Exception):
            self.finished = True
            self.path = []

    def _get_edge_len(self, u, v):
        if self.G.has_edge(u, v):
            data = self.G.get_edge_data(u, v)
            return data[0]['length']
        return 50.0

    def calculate_acceleration(self, gap, leader_velocity, dt):
        noise = 0.0
        if random.random() < 0.1: 
             noise = -random.uniform(0.0, 1.5)
        
        delta_v = self.velocity - leader_velocity
        s_star = self.s0 + max(0, self.velocity * self.T + (self.velocity * delta_v) / (2 * math.sqrt(self.a_max * self.b_comf)))
        
        effective_gap = max(0.1, gap)
        accel = self.a_max * (1 - (self.velocity / self.max_velocity)**self.delta - (s_star / effective_gap)**2)
        return accel + noise

    def update_physics(self, dt, traffic_sim):
        if self.finished: return

        current_u = self.path[self.path_index]
        current_v = self.path[self.path_index + 1]
        edge_key = (current_u, current_v)
        
        cars_here = traffic_sim.cars_on_edge.get(edge_key, [])
        try:
            my_idx = cars_here.index(self)
        except ValueError:
            my_idx = -1

        gap = 1000.0 
        leader_vel = self.max_velocity 
        max_progress_constraint = self.current_edge_len + 10.0 

        # STACKING FIX: Increased gap to 8.0m (~25px)
        SAFE_GAP = 8.0

        if my_idx > 0:
            leader = cars_here[my_idx - 1]
            gap = leader.progress - self.progress - leader.length
            leader_vel = leader.velocity
            max_progress_constraint = leader.progress - leader.length - SAFE_GAP 
        else:
            # Ghost Leader Logic
            dist_to_end = self.current_edge_len - self.progress
            gap = 1000.0
            leader_vel = self.max_velocity
            
            found_ghost = False
            if self.path_index < len(self.path) - 2:
                 next_node = self.path[self.path_index + 2]
                 next_edge_key = (current_v, next_node)
                 next_cars = traffic_sim.cars_on_edge.get(next_edge_key, [])
                 if next_cars:
                      ghost_leader = next_cars[-1] 
                      gap = dist_to_end + ghost_leader.progress - ghost_leader.length
                      leader_vel = ghost_leader.velocity
                      max_progress_constraint = self.current_edge_len + ghost_leader.progress - ghost_leader.length - SAFE_GAP
                      found_ghost = True

            light_state = traffic_sim.get_light_state(current_v)
            busy = traffic_sim.is_intersection_busy(current_v)
            
            if (light_state == 'RED' or busy) and not found_ghost:
                gap = dist_to_end
                leader_vel = 0.0
                max_progress_constraint = self.current_edge_len - SAFE_GAP
            elif (light_state == 'RED' or busy) and found_ghost:
                 stop_line_gap = dist_to_end
                 if stop_line_gap < gap:
                      gap = stop_line_gap
                      leader_vel = 0.0
                      max_progress_constraint = self.current_edge_len - SAFE_GAP

        self.acceleration = self.calculate_acceleration(gap, leader_vel, dt)
        self.velocity += self.acceleration * dt
        if self.velocity < 0: self.velocity = 0
            
        step_dist = self.velocity * dt + 0.5 * self.acceleration * (dt**2)
        if step_dist < 0: step_dist = 0
        
        target_progress = self.progress + step_dist
        
        # Apply Hard Constraint
        if target_progress > max_progress_constraint:
            target_progress = max_progress_constraint
            self.velocity = 0.0 

        if target_progress >= self.current_edge_len:
            # Gridlock Prevention
            can_enter_grid = True
            if self.path_index < len(self.path) - 2:
                next_node = self.path[self.path_index + 2]
                next_edge_key = (current_v, next_node)
                cars_ahead = traffic_sim.cars_on_edge.get(next_edge_key, [])
                if cars_ahead:
                    last_car = cars_ahead[-1] 
                    if last_car.progress < (last_car.length + SAFE_GAP):
                        can_enter_grid = False

            if not can_enter_grid:
                 self.progress = self.current_edge_len - 0.5
                 self.velocity = 0.0
            elif traffic_sim.try_enter_intersection(current_v):
                 if self.path_index < len(self.path) - 2:
                    self.path_index += 1
                    overhang = target_progress - self.current_edge_len
                    self.progress = overhang
                    u = self.path[self.path_index]
                    v = self.path[self.path_index + 1]
                    self.current_edge_len = self._get_edge_len(u, v)
                    traffic_sim.mark_intersection_busy(current_v)
                 else:
                    self.finished = True
                    self.velocity = 0
            else:
                 self.progress = self.current_edge_len - 0.1
                 self.velocity = 0
        else:
            self.progress = target_progress

    def get_state(self):
        # ... get_state remains as is (visuals are perfect) ...
        # But we need to include it because replace_file_content replaces chunks. 
        # Actually I can just skip it if I use Careful placement. But I must include standard code if replacing Update_Physics. 
        # I'll just copy the existing visual code logic here to ensure it's not lost.
        
        color = (20, 200, 20) 
        if self.velocity < 2.0:
             color = (255, 30, 30) 
        elif self.velocity < 7.0: 
             color = (255, 140, 0) 

        if self.finished or not self.path:
             return 0, 0, 0, False, color

        u = self.path[self.path_index]
        v = self.path[self.path_index + 1]
        
        data = self.G.get_edge_data(u, v)[0]
        
        current_x, current_y = 0, 0
        current_angle = 0
        
        geom_visual = data.get('geometry_visual', None)
        
        if geom_visual:
             original_len = data['length']
             visual_len = geom_visual.length
             if original_len > 0:
                 ratio = self.progress / original_len
                 visual_progress = ratio * visual_len
             else:
                 visual_progress = 0
                 
             p_val = max(0.0, min(visual_progress, visual_len))
             pt = geom_visual.interpolate(p_val)
             current_x, current_y = pt.x, pt.y
             
             p_next = max(0.0, min(visual_progress + 1.0, visual_len))
             pt2 = geom_visual.interpolate(p_next)
             dx = pt2.x - pt.x
             dy = pt2.y - pt.y
             if math.hypot(dx, dy) < 0.01:
                  p_prev = max(0.0, min(visual_progress - 1.0, visual_len))
                  pt_prev = geom_visual.interpolate(p_prev)
                  dx = pt.x - pt_prev.x
                  dy = pt.y - pt_prev.y
             current_angle = math.degrees(math.atan2(dy, dx))
        else:
            u_node = self.G.nodes[u]
            v_node = self.G.nodes[v]
            dx = v_node['x'] - u_node['x']
            dy = v_node['y'] - u_node['y']
            current_angle = math.degrees(math.atan2(dy, dx))
            ratio = min(1.0, self.progress / self.current_edge_len)
            current_x = u_node['x'] + dx * ratio
            current_y = u_node['y'] + dy * ratio
        
        return current_x, current_y, current_angle, False, color

class TrafficSim:
    def __init__(self, G):
        self.G = G
        self.cars = []
        self.nodes = list(G.nodes())
        self.node_locks = {n: 0 for n in self.nodes}
        self.cars_on_edge = {}
        self.traffic_lights = {}
        self._init_traffic_lights()
        self.entry_nodes = []
        self._identify_entry_nodes()

    def _identify_entry_nodes(self):
        # Calculate bounds
        xs = [d['x'] for n, d in self.G.nodes(data=True)]
        ys = [d['y'] for n, d in self.G.nodes(data=True)]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        buffer = 0.1 # 10% edge buffer
        
        self.entry_nodes = []
        self.hidden_edges = set()
        
        for n, d in self.G.nodes(data=True):
            # Check if near edges
            if (d['x'] < min_x + width*buffer) or \
               (d['x'] > max_x - width*buffer) or \
               (d['y'] < min_y + height*buffer) or \
               (d['y'] > max_y - height*buffer):
                   self.entry_nodes.append(n)
        
        # Identify Hidden Edges (Connected to Entry Nodes)
        for u, v, k in self.G.edges(keys=True):
            if u in self.entry_nodes or v in self.entry_nodes:
                self.hidden_edges.add((u, v))
                
        print(f"Identified {len(self.entry_nodes)} Entry Nodes and {len(self.hidden_edges)} Hidden Edges.", flush=True)

    def _init_traffic_lights(self):
        degrees = dict(self.G.degree())
        for node in self.nodes:
            if degrees.get(node, 0) >= 3:
                self.traffic_lights[node] = TrafficLight(node, cycle_time=random.uniform(20.0, 40.0))

    def get_light_state(self, node):
        if node in self.traffic_lights:
            return 'GREEN' if self.traffic_lights[node].is_green else 'RED'
        return 'GREEN'

    def spawn_cars(self, n):
        for i in range(n):
            self.add_car(i)

    def _noisy_weight(self, u, v, d):
        # Noise Factor: 0.5x to 5.0x length (Increased variance to force loop usage)
        return d[0]['length'] * random.uniform(0.5, 5.0)

    def get_random_path(self, origin, dest):
        try:
             return nx.shortest_path(self.G, origin, dest, weight=self._noisy_weight)
        except:
             return None

    def add_car(self, i):
        if not self.entry_nodes:
             origin = random.choice(self.nodes)
        else:
             origin = random.choice(self.entry_nodes) # SPAWN FROM EDGE
        
        dest = random.choice(self.nodes)
        while origin == dest:
             dest = random.choice(self.nodes)
             
        # Calculate path with NOISE here
        path = self.get_random_path(origin, dest)
        if path:
            car = Car(i, origin, dest, self.G)
            car.reset(origin, dest, path) 
            self.cars.append(car)
            # If path failed in reset (shouldn't if get_random_path worked), it might be finished immediately
            if car.finished: 
                self.cars.pop() # Remove bad car immediately
        else:
            # Failed to find path, don't add car
            pass

    def reset_car_route(self, car):
        if not self.entry_nodes:
             origin = random.choice(self.nodes)
        else:
             origin = random.choice(self.entry_nodes)
             
        dest = random.choice(self.nodes)
        while origin == dest:
             dest = random.choice(self.nodes)
        
        path = self.get_random_path(origin, dest)
        if path:
             car.reset(origin, dest, path)
        else:
             self.reset_car_route(car) # Retry

    def is_intersection_busy(self, node):
        return self.node_locks[node] > 0

    def try_enter_intersection(self, node):
        return self.node_locks[node] <= 0

    def mark_intersection_busy(self, node):
        self.node_locks[node] = 10 

    def step(self, dt):
        for light in self.traffic_lights.values():
            light.update(dt)
        for n in self.node_locks:
            if self.node_locks[n] > 0:
                self.node_locks[n] -= 1
        
        # OBJECT PERMANENCE & RESPAWN
        for car in self.cars:
            if car.finished:
                self.reset_car_route(car) 
                
        self.cars_on_edge = {}
        # Filter active cars strictly
        active_cars = [c for c in self.cars if not c.finished and c.path]
        
        for car in active_cars:
            u = car.path[car.path_index]
            v = car.path[car.path_index + 1]
            edge = (u, v)
            if edge not in self.cars_on_edge:
                self.cars_on_edge[edge] = []
            self.cars_on_edge[edge].append(car)
        for edge in self.cars_on_edge:
            self.cars_on_edge[edge].sort(key=lambda c: c.progress, reverse=True)
            
        for car in active_cars:
            car.update_physics(dt, self)
            
        # Re-filter in case physics finished them this frame (Prevent (0,0) glitch)
        final_active_cars = [c for c in active_cars if not c.finished]
        return final_active_cars

# VISUAL HELPERS
def world_to_screen(wx, wy, cam_x, cam_y, zoom, base_scale, screen_h):
    # World: x, y (meters, y up)
    # Screen: x, y (pixels, y down)
    
    # Apply Zoom & Camera Pan
    # cam_x, cam_y is the center of the camera in WORLD coords? NO, typically camera offset in pixels
    # Let's say cam_x, cam_y is translation in pixels.
    
    sx = wx * base_scale * zoom + cam_x
    sy = screen_h - (wy * base_scale * zoom + cam_y) # Flip Y
    return int(sx), int(sy)

def main():
    print("Initializing Realistic PyGame Simulation...", flush=True)
    point = (44.478121, 26.072711)
    G = ox.graph_from_point(point, dist=400, network_type='drive')
    G_proj = ox.project_graph(G)
    
    print("Pre-calculating Visual Geometries...", flush=True)
    offset_dist = 6.0
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        if 'geometry' in data:
            raw_geom = data['geometry']
            try:
                # Calculate Offset
                vis_geom = raw_geom.parallel_offset(offset_dist, 'right', resolution=16, join_style=2)
                
                # Handling Geometry Types
                if vis_geom.is_empty:
                     data['geometry_visual'] = raw_geom # Fallback
                elif vis_geom.geom_type == 'LineString':
                     # parallel_offset usually (but not always) preserves direction for 'right'. 
                     # If it looks reversed, we might need check. 
                     # Simple check: distance(start, offset_start) vs distance(start, offset_end)
                     p0 = raw_geom.interpolate(0)
                     vp0 = vis_geom.interpolate(0)
                     vp_end = vis_geom.interpolate(vis_geom.length)
                     
                     d_normal = p0.distance(vp0)
                     d_flipped = p0.distance(vp_end)
                     
                     if d_flipped < d_normal: # It's reversed
                          vis_geom = LineString(list(vis_geom.coords)[::-1])
                          
                     data['geometry_visual'] = vis_geom
                elif vis_geom.geom_type == 'MultiLineString':
                     # Pick longest
                     best_g = max(vis_geom.geoms, key=lambda g: g.length)
                     data['geometry_visual'] = best_g
                else:
                     data['geometry_visual'] = raw_geom
            except:
                data['geometry_visual'] = raw_geom
        else:
             # Create Logic for Straight Edge
             u_node = G_proj.nodes[u]
             v_node = G_proj.nodes[v]
             ls = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
             try:
                 vis_geom = ls.parallel_offset(offset_dist, 'right', resolution=2)
                 if vis_geom.is_empty: data['geometry_visual'] = ls
                 else: 
                     # Check reverse
                     p0 = ls.interpolate(0)
                     vp0 = vis_geom.interpolate(0)
                     vp_end = vis_geom.interpolate(vis_geom.length)
                     if p0.distance(vp_end) < p0.distance(vp0):
                         vis_geom = LineString(list(vis_geom.coords)[::-1])
                     data['geometry_visual'] = vis_geom
             except:
                 data['geometry_visual'] = ls

    sim = TrafficSim(G_proj)
    print("Spawning 200 cars...", flush=True)
    sim.spawn_cars(200)

    # Calculate Bounds
    nodes = list(G_proj.nodes(data=True))
    xs = [d['x'] for n, d in nodes]
    ys = [d['y'] for n, d in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width_m = max_x - min_x
    height_m = max_y - min_y
    
    # VISUAL STAGE DEFINITION
    # Hide the outer 10% (Wings) so cars appear to enter/exit from off-screen
    # This also naturally hides any (0,0) glitches or far-flung artifacts
    stage_buffer_x = width_m * 0.08 
    stage_buffer_y = height_m * 0.08
    
    stage_min_x = min_x + stage_buffer_x
    stage_max_x = max_x - stage_buffer_x
    stage_min_y = min_y + stage_buffer_y
    stage_max_y = max_y - stage_buffer_y

    def is_on_stage(wx, wy):
        return (stage_min_x <= wx <= stage_max_x) and (stage_min_y <= wy <= stage_max_y)

    # Init PyGame
    pygame.init()
    SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Traffic Sim: Scroll to Zoom, Drag to Pan")
    clock = pygame.time.Clock()

    # Camera State
    zoom = 1.0
    # Center map initially
    padding = 50
    scale_x = (SCREEN_WIDTH - 2*padding) / width_m
    scale_y = (SCREEN_HEIGHT - 2*padding) / height_m
    base_scale = min(scale_x, scale_y)
    
    # Centering translation
    cx_world = width_m/2
    cy_world = height_m/2
    
    cam_x = SCREEN_WIDTH/2 - cx_world * base_scale
    cam_y = SCREEN_HEIGHT/2 - cy_world * base_scale 
    
    dragging = False
    last_mouse = (0, 0)

    # GIF Recording
    is_recording = False
    recorded_frames = []

    running = True
    print("Starting Loop...", flush=True)
    
    font = pygame.font.SysFont("arial", 20)

    while running:
        dt = 0.1 # Fixed sim step
        
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4: # Scroll Up -> Zoom In
                    zoom *= 1.1
                elif event.button == 5: # Scroll Down -> Zoom Out
                    zoom /= 1.1
                elif event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 3: # Right click record
                    if not is_recording:
                         is_recording = True
                         recorded_frames = []
                         print("Recording...")
                    else:
                         is_recording = False
                         print("Saving GIF...")
                         if recorded_frames:
                             pil_images = [PIL.Image.frombytes('RGB', screen.get_size(), f) for f in recorded_frames]
                             pil_images[0].save(f"sim_{int(time.time())}.gif", save_all=True, append_images=pil_images[1:], duration=66, loop=0)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    cam_x += dx
                    cam_y -= dy 
                    last_mouse = event.pos
            elif event.type == pygame.VIDEORESIZE:
                SCREEN_WIDTH, SCREEN_HEIGHT = event.w, event.h

        # Update Sim
        active_cars = sim.step(dt)

        # Draw
        screen.fill((20, 30, 70)) # Dark Blue
        
        # Helper for transforms
        def to_scr(px, py):
            # px, py are World Coords (relative to min_x, min_y)
            sx = px * base_scale * zoom + cam_x
            sy = SCREEN_HEIGHT - (py * base_scale * zoom + cam_y)
            return int(sx), int(sy)

        # Draw Roads
        road_width = int(30 * zoom)
        if road_width < 1: road_width = 1
        
        for u, v, k, data in G_proj.edges(keys=True, data=True):
            # VISUAL MASK: Clip roads strictly to stage
            # We don't hide edges list logic, we use geometric logic
            # If both nodes are off stage, skip
            u_node = G_proj.nodes[u]
            v_node = G_proj.nodes[v]
            
            # Liberal check for roads (draw if partially visible)
            # Actually, to be clean, if EITHER is off-stage, we might want to hide? 
            # User wants "Entering" effect. 
            # If we hide the road, cars floating on it will look weird unless we hide cars too.
            # We hide cars strictly. We can hide roads strictly too (both must be on stage?)
            # Let's hide road if BOTH are off-stage.
            if not is_on_stage(u_node['x'], u_node['y']) and not is_on_stage(v_node['x'], v_node['y']):
                 continue

            geom = data.get('geometry_visual', None)
            if geom:
                try:
                    xs, ys = geom.xy
                    pts = []
                    for x, y in zip(xs, ys):
                        pts.append(to_scr(x - min_x, y - min_y))
                    
                    if len(pts) > 1:
                        pygame.draw.lines(screen, (60, 60, 70), False, pts, road_width)
                        if road_width > 5:
                            pygame.draw.lines(screen, (200, 200, 200), False, pts, 1)
                except: pass

        # Draw Nodes (Intersections)
        node_radius = int((road_width/2) * 1.2)
        for n, d in G_proj.nodes(data=True):
             # Strict Mask for Nodes
             if not is_on_stage(d['x'], d['y']):
                 continue
                 
             pos = to_scr(d['x'] - min_x, d['y'] - min_y)
             colors = (60, 60, 70)
             if n in sim.traffic_lights:
                 l = sim.traffic_lights[n]
                 colors = (0, 255, 0) if l.is_green else (255, 0, 0)
                 
             pygame.draw.circle(screen, colors, pos, node_radius)

        # Draw Cars
        car_w = int(12 * zoom)
        car_h = int(6 * zoom)
        if car_w < 2: car_w = 2
        if car_h < 1: car_h = 1

        for car in active_cars:
             cx, cy, angle, _, color = car.get_state()
             
             # STRICT GEOMETRIC MASK
             # If car is geometrically in the "Wings" (outer margin), DO NOT DRAW
             if not is_on_stage(cx, cy):
                 continue

             sx, sy = to_scr(cx - min_x, cy - min_y)
             
             car_surf = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
             car_surf.fill(color)
             
             rotated_surf = pygame.transform.rotate(car_surf, angle)
             rect = rotated_surf.get_rect(center=(sx, sy))
             screen.blit(rotated_surf, rect)

        # UI Overlay
        ui_txt = font.render(f"Zoom: {zoom:.1f} | Cars: {len(active_cars)}", True, (255, 255, 255))
        screen.blit(ui_txt, (10, 10))
        
        if is_recording:
             rec_txt = font.render("REC", True, (255, 0, 0))
             screen.blit(rec_txt, (SCREEN_WIDTH-50, 10))
             # Capture
             if frame_counter % 4 == 0:
                  from pygame.image import tostring
                  # Capture efficient
                  frame_str = tostring(screen, 'RGB')
                  recorded_frames.append(frame_str)
             frame_counter += 1

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
