# ImageEvo - Evolutionary Image Approximation

![Evolution Example](example_output.png)  
*Evolution progress after 10,000 generations (target: abbey_road.png)*
 
## Description
ImageEvo is a genetic algorithm that evolves images using polygons to approximate a target image. It employs advanced parallel processing techniques and visual quality metrics to optimize the evolution process.

## Key Features

### 🧬 Advanced Evolutionary Algorithm
- **Polygon-based representation**: Generates images through geometric shape composition
- **Adaptive mutation**: Adjusts intensity based on current generation
- **Tournament selection**: With elitism (preserves top 15% individuals)
- **Population diversity**: Color histogram-based metric to prevent premature convergence

### 📊 Quality Metrics
- **Delta E CIE2000**: Precise color difference measurement
- **SSIM (Structural Similarity)**: Image structure evaluation
- **Multi-scale analysis**: Evaluates at different resolutions for better accuracy

### ⚡ Optimizations
- **Parallel processing**: Utilizes all CPU cores for fitness evaluation
- **Real-time GUI**: Tkinter + Matplotlib visualization
- **Smart post-processing**: Gaussian sharpening for enhanced display

## Requirements
```bash
pip install -r requirements.txt
