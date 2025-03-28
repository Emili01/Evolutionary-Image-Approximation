"""
EVOLUTIONARY IMAGE APPROXIMATION ALGORITHM
Uses genetic programming to evolve images toward a target using:
- Polygon-based mutations
- Parallel fitness evaluation
- Adaptive evolutionary strategies
"""

# ==================== IMPORTS ====================
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import random
import tkinter as tk
from tkinter import ttk
from colour.difference import delta_E_CIE2000  # More accurate color difference metric
from multiprocessing import Pool, cpu_count    # Parallel processing
import time
from skimage.metrics import structural_similarity as ssim  # Image structure comparison
from skimage.color import rgb2lab, rgb2gray
import matplotlib.pyplot as plt               # Progress visualization
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
from scipy.ndimage import gaussian_filter    # For image sharpening

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==================== INDIVIDUAL CLASS ====================
class Individual:
    """Represents a candidate solution (image) in the population"""
    
    def __init__(self, width, height, generation=0, image=None, array=None):
        self.width = width
        self.height = height
        self.generation = generation
        self.fitness = float('inf')          # Initial fitness (lower is better)
        self.ssim_score = -1.0               # Structural similarity score
        self.image = image if image else self._create_random_image()
        self.array = array if array is not None else np.array(self.image)
        
    def _random_color(self):
        """Generates random RGBA color with full opacity"""
        return tuple(random.randint(0, 255) for _ in range(3)) + (255,)
    
    def _create_random_image(self):
        """Creates initial image with random polygons"""
        img = Image.new("RGBA", (self.width, self.height), self._random_color())
        draw = ImageDraw.Draw(img)
        
        # Dynamic polygon count based on image size
        min_polygons = max(3, int((self.width * self.height) ** 0.5 / 20))
        max_polygons = min(10, min_polygons * 2)
        
        for _ in range(random.randint(min_polygons, max_polygons)):
            region = min(self.width, self.height) // 4
            center_x = random.randint(0, self.width)
            center_y = random.randint(0, self.height)
            
            # Create semi-regular polygons
            num_points = random.randint(3, 6)
            radius = random.randint(region//4, region)
            points = []
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                # Add some randomness to polygon shape
                x = center_x + radius * np.cos(angle) * (0.8 + random.random() * 0.4)
                y = center_y + radius * np.sin(angle) * (0.8 + random.random() * 0.4)
                points.append((x, y))
            
            draw.polygon(points, fill=self._random_color())
        
        return img
    
    def mutate(self, current_generation):
        """Applies adaptive mutation based on current generation"""
        draw = ImageDraw.Draw(self.image)
        
        # Mutation strength decreases over generations
        mutation_strength = max(1, 5 - current_generation//300)
        region = min(self.width, self.height) // (4 + mutation_strength)
        
        for _ in range(mutation_strength):
            mutation_type = random.random()
            
            # 70% chance of polygon mutation
            if mutation_type < 0.7:
                center_x = random.randint(0, self.width)
                center_y = random.randint(0, self.height)
                
                num_points = random.randint(3, 5)
                radius = random.randint(region//2, region)
                points = []
                
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    x = center_x + radius * np.cos(angle) * (0.7 + random.random() * 0.6)
                    y = center_y + radius * np.sin(angle) * (0.7 + random.random() * 0.6)
                    points.append((x, y))
                
                draw.polygon(points, fill=self._random_color())
            # 30% chance of rectangle color mutation
            else:
                x1 = random.randint(0, self.width)
                y1 = random.randint(0, self.height)
                x2 = min(self.width, x1 + random.randint(10, region))
                y2 = min(self.height, y1 + random.randint(10, region))
                draw.rectangle([x1, y1, x2, y2], fill=self._random_color())
        
        # Update internal representation
        self.array = np.array(self.image)
        self.generation = current_generation

# ==================== MAIN APPLICATION CLASS ====================
class ImageEvolverApp:
    """TKinter application managing the evolutionary process"""
    
    def __init__(self, root, target_path, width=200, height=200):
        self.root = root
        self.width = width
        self.height = height
        self.generation = 0
        self.running = False
        self.last_update_time = time.time()
        
        # Evolutionary parameters
        self.use_ssim = True                # Whether to use structural similarity
        self.mix_ratio = 0.7                # Weight between SSIM and color difference
        self.pop_size = 120                 # Population size
        self.elitism_rate = 0.15            # Percentage of best individuals to keep
        self.mutation_rate = 0.3            # Base mutation probability
        self.tournament_size = 7            # Selection tournament size
        self.diversity_threshold = 0.05     # Threshold for fitness sharing
        self.generation_limit = 10000       # Stopping condition

        # Tracking progress
        self.fitness_history = []
        self.ssim_history = []
        self.best_images = []
        self.diversity_history = []
        
        # Parallel processing pool
        self.pool = Pool(processes=max(1, cpu_count() - 1))
        
        # Load and preprocess target image
        self.target = Image.open(target_path).convert('RGBA').resize(
            (width, height), Image.Resampling.LANCZOS)
        self.target_array = np.array(self.target)
        
        # Multi-scale target versions for fitness calculation
        self.target_small = self.target_array[::4, ::4]  # 4x downsampled
        self.target_medium = self.target_array[::2, ::2]  # 2x downsampled
        
        self.setup_gui()
        self.initialize_population(self.pop_size)
    
    def setup_gui(self):
        self.root.title("Image Evolution")
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_evolved = tk.Canvas(img_frame, width=self.width, height=self.height)
        self.canvas_evolved.pack(side=tk.LEFT, padx=5)
        ttk.Label(img_frame, text="Best Individual").pack(side=tk.LEFT)
        
        self.canvas_target = tk.Canvas(img_frame, width=self.width, height=self.height)
        self.canvas_target.pack(side=tk.RIGHT, padx=5)
        ttk.Label(img_frame, text="Target").pack(side=tk.RIGHT)
        
        self.target_tk = ImageTk.PhotoImage(self.target)
        self.canvas_target.create_image(0, 0, anchor=tk.NW, image=self.target_tk)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.btn_start = ttk.Button(control_frame, text="Start", command=self.start_evolution)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Stop", command=self.stop_evolution)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="SSIM:").pack(side=tk.LEFT, padx=5)
        self.ssim_var = tk.BooleanVar(value=True)
        self.chk_ssim = ttk.Checkbutton(control_frame, variable=self.ssim_var)
        self.chk_ssim.pack(side=tk.LEFT)
        
        ttk.Label(control_frame, text="Mix:").pack(side=tk.LEFT, padx=5)
        self.scale_mix = ttk.Scale(control_frame, from_=0, to=100, value=70)
        self.scale_mix.pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = ttk.Label(control_frame, 
                                  text="Generation: 0 | Fitness: ∞ | SSIM: - | FPS: 0")
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.frame_count = 0
        self.fps = 0
        self.last_diversity_check = 0
    
    def initialize_population(self, size):
        args = [(self.width, self.height) for _ in range(size)]
        results = self.pool.starmap(create_individual, args)
        self.population = results
        
        for i in range(min(5, size//10)):
            self.population[i] = self.create_gradient_individual()
    
    def create_gradient_individual(self):
        img = Image.new("RGBA", (self.width, self.height))
        draw = ImageDraw.Draw(img)
        
        target_colors = self.target_array.reshape(-1, 4)
        unique_colors = np.unique(target_colors, axis=0)
        if len(unique_colors) > 2:
            color1, color2 = unique_colors[0], unique_colors[-1]
        else:
            color1, color2 = (0, 0, 0, 255), (255, 255, 255, 255)
        
        for x in range(self.width):
            ratio = x / self.width
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(x, 0), (x, self.height)], fill=(r, g, b, 255))
        
        return Individual(self.width, self.height, 0, img, np.array(img))
    
    def start_evolution(self):
        if not self.running:
            self.running = True
            self.evolve()
    
    def stop_evolution(self):
        self.running = False
        self.pool.close()
        self.pool.join()
    
    def evolve(self):
        if not self.running or self.generation >= self.generation_limit:
            self.stop_evolution()
            return
        
        start_time = time.time()
        self.generation += 1
        
        self.mix_ratio = self.scale_mix.get() / 100.0
        self.use_ssim = self.ssim_var.get()
        
        fitness_args = [(ind.array, self.target_small, self.target_medium, 
                       self.use_ssim, self.mix_ratio) for ind in self.population]
        fitness_results = self.pool.starmap(calculate_fitness_parallel, fitness_args)
        
        for ind, (fitness, ssim_score) in zip(self.population, fitness_results):
            ind.fitness = fitness
            ind.ssim_score = ssim_score
        
        parents = []
        for _ in range(len(self.population)):
            candidates = random.sample(self.population, self.tournament_size)
            best = min(candidates, key=lambda x: x.fitness)
            
            similar_count = sum(1 for c in candidates 
                              if abs(c.fitness - best.fitness) < self.diversity_threshold)
            if similar_count > 1:
                best = random.choice([c for c in candidates 
                                    if abs(c.fitness - best.fitness) < self.diversity_threshold])
            
            parents.append(best)
        
        elite_size = int(self.elitism_rate * len(self.population))
        elite = sorted(self.population, key=lambda x: x.fitness)[:elite_size]
        
        child_args = [(parent1, parent2, self.width, self.height, self.generation)
                     for parent1, parent2 in zip(parents, parents[1:] + parents[:1])]
        children = self.pool.starmap(create_child, child_args)
        
        diversity = self.calculate_population_diversity()
        self.diversity_history.append(diversity)
        
        for child in children:
            current_mutation_rate = min(0.5, self.mutation_rate + (0.3 if diversity < 0.1 else 0))
            
            if random.random() < current_mutation_rate:
                child.mutate(self.generation)
                child.fitness, child.ssim_score = calculate_fitness_parallel(
                    child.array, self.target_small, self.target_medium, 
                    self.use_ssim, self.mix_ratio)
        
        self.population = elite + children[:len(self.population)-elite_size]
        
        best = min(self.population, key=lambda x: x.fitness)
        self.fitness_history.append(best.fitness)
        self.ssim_history.append(best.ssim_score)
        self.best_images.append(best.image.copy())
        
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_update_time >= 0.5:
            self.fps = self.frame_count / (current_time - self.last_update_time)
            self.frame_count = 0
            self.last_update_time = current_time
            
            self.update_display(best)
            self.update_plots()
            self.lbl_status.config(
                text=f"Gen: {self.generation} | Fit: {best.fitness:.4f} | " +
                     f"SSIM: {best.ssim_score:.3f} | Div: {diversity:.3f} | FPS: {self.fps:.1f}")
        
        elapsed = time.time() - start_time
        delay = max(1, int(50 - elapsed * 1000))
        
        self.root.after(delay, self.evolve)
    
    def calculate_population_diversity(self):
        if len(self.population) < 2:
            return 1.0
        
        histograms = []
        for ind in self.population:
            img = ind.image.convert("RGB")
            hist = np.concatenate([
                np.array(img.histogram()[0:256]),
                np.array(img.histogram()[256:512]),
                np.array(img.histogram()[512:768])
            ])
            hist = hist / (self.width * self.height)
            histograms.append(hist)
        
        diffs = []
        for i in range(len(histograms)):
            for j in range(i+1, len(histograms)):
                diffs.append(np.sum(np.abs(histograms[i] - histograms[j])))
        
        return np.mean(diffs) if diffs else 0
    
    def check_convergence(self):
        if len(self.fitness_history) < 200:
            return False
        
        best_fitness = min(self.fitness_history)
        if best_fitness < 0.03:
            return True
        
        window = 100
        if len(self.fitness_history) > window:
            recent_improvement = abs(np.mean(self.fitness_history[-window//2:]) - 
                                   np.mean(self.fitness_history[-window:-window//2]))
            return recent_improvement < 0.001 and best_fitness < 0.1
        
        return False
    
    def update_display(self, individual):
        img = individual.image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        blurred = gaussian_filter(img_array, sigma=0.5)
        sharpened = np.clip(img_array * 1.3 - blurred * 0.3, 0, 255).astype(np.uint8)
        img = Image.fromarray(sharpened)
        
        self.evolved_tk = ImageTk.PhotoImage(img)
        self.canvas_evolved.delete("all")
        self.canvas_evolved.create_image(0, 0, anchor=tk.NW, image=self.evolved_tk)
    
    def update_plots(self):
        self.ax1.clear()
        self.ax1.plot(self.fitness_history, 'b-')
        self.ax1.set_title("Fitness Evolution")
        self.ax1.set_ylabel("Fitness (lower is better)")
        self.ax1.grid(True)
        
        self.ax2.clear()
        self.ax2.plot(self.ssim_history, 'r-')
        self.ax2.set_title("SSIM Evolution")
        self.ax2.set_ylabel("SSIM (higher is better)")
        self.ax2.grid(True)
        
        self.ax3.clear()
        self.ax3.plot(self.diversity_history, 'g-')
        self.ax3.set_title("Population Diversity")
        self.ax3.set_ylabel("Diversity")
        self.ax3.grid(True)
        
        self.canvas.draw()

# ==================== PARALLEL PROCESSING FUNCTIONS ====================
def create_individual(width, height):
    """Wrapper function for parallel individual creation"""
    return Individual(width, height)


def calculate_fitness_parallel(array, target_small, target_medium, use_ssim, mix_ratio):
    """
    Parallel fitness calculation using:
    - Delta E 2000 for color difference
    - SSIM for structural similarity
    - Multi-scale evaluation (4x and 2x downsampling)
    """
    small_array = array[::4, ::4]
    small_array = small_array[:, :, :3] if small_array.shape[2] == 4 else small_array
    small_target = target_small[:, :, :3] if target_small.shape[2] == 4 else target_small
    
    medium_array = array[::2, ::2]
    medium_array = medium_array[:, :, :3] if medium_array.shape[2] == 4 else medium_array
    medium_target = target_medium[:, :, :3] if target_medium.shape[2] == 4 else target_medium
    
    delta_e_small = np.mean(delta_E_CIE2000(small_target, small_array))
    delta_e_medium = np.mean(delta_E_CIE2000(medium_target, medium_array))
    delta_e = 0.6 * delta_e_small + 0.4 * delta_e_medium
    
    if use_ssim:
        gray_small_array = rgb2gray(small_array)
        gray_small_target = rgb2gray(small_target)
        ssim_small = ssim(gray_small_target, gray_small_array, 
                         data_range=1.0, win_size=3, channel_axis=None)
        
        gray_medium_array = rgb2gray(medium_array)
        gray_medium_target = rgb2gray(medium_target)
        ssim_medium = ssim(gray_medium_target, gray_medium_array,
                          data_range=1.0, win_size=3, channel_axis=None)
        
        ssim_val = 0.7 * ssim_small + 0.3 * ssim_medium
        
        fitness = (mix_ratio * (1 - ssim_val)) + ((1 - mix_ratio) * (delta_e/100.0))
        return (fitness, ssim_val)
    else:
        return (delta_e, -1)

def create_child(parent1, parent2, width, height, generation):
    """
    Creates offspring through multiple crossover strategies:
    1. Alpha blending (70% chance)
    2. Region-based crossover (20% chance)
    3. Channel-wise blending (10% chance)
    """
    
    blend_type = random.random()
    
    if blend_type < 0.7:
        alpha = random.random()
        child_array = (parent1.array * alpha + parent2.array * (1 - alpha)).astype(np.uint8)
    elif blend_type < 0.9:
        mask = np.random.random((height, width, 1)) > 0.5
        child_array = np.where(mask, parent1.array, parent2.array)
    else:
        alpha_r = random.random()
        alpha_g = random.random()
        alpha_b = random.random()
        child_array = np.stack([
            (parent1.array[:,:,0] * alpha_r + parent2.array[:,:,0] * (1 - alpha_r)),
            (parent1.array[:,:,1] * alpha_g + parent2.array[:,:,1] * (1 - alpha_g)),
            (parent1.array[:,:,2] * alpha_b + parent2.array[:,:,2] * (1 - alpha_b)),
            np.full((height, width), 255)
        ], axis=-1).astype(np.uint8)
    
    child_image = Image.fromarray(child_array)
    return Individual(width, height, generation, child_image, np.array(child_image))



# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEvolverApp(root, "abbey_road.png") # Change to your target image
    root.mainloop()
