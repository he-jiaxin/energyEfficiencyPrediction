import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image
from pdf2image import convert_from_path

class HeatmapVisualizer:
    def __init__(self, layout_img_path, mask_img_path):
        self.layout_img_path = layout_img_path
        self.mask_img_path = mask_img_path
        self.layout_img = self.load_image(self.layout_img_path)
        self.mask_img = self.load_image(self.mask_img_path).rotate(-90, expand=True)
        self.mask_binary_resized = self.process_mask_with_morphology()
        self.window_dist = self.calculate_window_distance()
        self.combined_heatmap = self.generate_map(is_heat=True)
        self.combined_coolmap = self.generate_map(is_heat=False)

    def load_image(self, img_path):
        if img_path.lower().endswith('.pdf'):
            images = convert_from_path(img_path)
            return images[0]
        else:
            return Image.open(img_path)

    def process_mask_with_morphology(self, tolerance=10):
        mask_array = np.array(self.mask_img)
        layout_size = self.layout_img.size

        room_colors = [
            [255, 224, 128],
            [224, 255, 192],
            [192, 255, 255],
            [255, 160, 96],
            [224, 224, 224],
            [224, 224, 128],
        ]

        mask_binary = np.zeros(mask_array.shape[:2], dtype=np.float32)
        for color in room_colors:
            color = np.array(color)
            color_mask = np.all(np.abs(mask_array - color) <= tolerance, axis=-1)
            mask_binary = np.logical_or(mask_binary, color_mask)

        mask_binary = mask_binary.astype(np.float32)

        mask_binary_resized = zoom(mask_binary, 
                                   [layout_size[1] / mask_binary.shape[0], 
                                    layout_size[0] / mask_binary.shape[1]], 
                                   order=0)

        return mask_binary_resized

    def calculate_window_distance(self):
        window_color = np.array([255, 60, 128])
        mask_array = np.array(self.mask_img)
        
        window_mask = np.all(np.abs(mask_array - window_color) <= 10, axis=-1)
        window_mask_resized = zoom(window_mask.astype(np.float32),
                                [self.layout_img.size[1] / mask_array.shape[0], 
                                    self.layout_img.size[0] / mask_array.shape[1]],
                                order=0)

        window_dist = np.ones_like(window_mask_resized) * np.inf
        window_indices = np.where(window_mask_resized > 0)
        for y, x in zip(*window_indices):
            y_coords, x_coords = np.ogrid[:window_dist.shape[0], :window_dist.shape[1]]
            distance = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            window_dist = np.minimum(window_dist, distance)

        return window_dist / np.max(window_dist)

    def generate_map(self, is_heat):
        combined_map = np.zeros(self.mask_binary_resized.shape)
        np.random.seed(42)
        room_indices = np.where(self.mask_binary_resized == 1)
        map_data = np.zeros_like(self.mask_binary_resized)
        map_data[room_indices] = np.random.beta(a=0.5, b=2, size=len(room_indices[0]))

        map_blurred = gaussian_filter(map_data, sigma=60)
        combined_map = map_blurred * 0.9 + self.window_dist * 0.1

        combined_map[self.mask_binary_resized == 0] = 0
        combined_map = (combined_map - np.min(combined_map)) / (np.max(combined_map) - np.min(combined_map))

        return combined_map

    def plot_heatmap(self):
        cmap_heat = plt.cm.jet
        cmap_heat.set_under('none')

        cmap_cool = plt.cm.jet_r
        cmap_cool.set_under('none')

        threshold = 0.02
        heatmap_with_threshold = np.where(self.combined_heatmap < threshold, np.nan, self.combined_heatmap)
        coolmap_with_threshold = np.where(self.combined_coolmap < threshold, np.nan, self.combined_coolmap)

        layout_img_array = np.array(self.layout_img)
        extent = (0, layout_img_array.shape[1], 0, layout_img_array.shape[0])

        # Create heatmap
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.imshow(layout_img_array, extent=extent, aspect='equal')
        heatmap_img = ax1.imshow(heatmap_with_threshold.T, extent=extent, origin='lower', cmap=cmap_heat, alpha=0.65, vmin=threshold)
        ax1.axis('off')
        cbar_heat = fig1.colorbar(heatmap_img, ax=ax1, orientation='vertical', shrink=0.8)
        cbar_heat.set_label('Heat Load')

        # Save the heatmap image
        output_dir = "/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output"
        base_name = os.path.splitext(os.path.basename(self.layout_img_path))[0]
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
        fig1.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)

        # Create coolmap
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.imshow(layout_img_array, extent=extent, aspect='equal')
        coolmap_img = ax2.imshow(coolmap_with_threshold.T, extent=extent, origin='lower', cmap=cmap_cool, alpha=0.65, vmin=threshold)
        ax2.axis('off')
        cbar_cool = fig2.colorbar(coolmap_img, ax=ax2, orientation='vertical', shrink=0.8)
        cbar_cool.set_label('Cooling Load')

        # Save the coolmap image
        coolmap_path = os.path.join(output_dir, f"{base_name}_coolmap.png")
        fig2.savefig(coolmap_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmap visualization.')
    parser.add_argument('--layout_img_path', type=str, required=True, help='Path to the layout image.')

    args = parser.parse_args()

    layout_img_path = args.layout_img_path
    base_name = os.path.splitext(os.path.basename(layout_img_path))[0]
    mask_img_path = os.path.join('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output', f"{base_name}_result.jpg")

    visualizer = HeatmapVisualizer(layout_img_path, mask_img_path)
    visualizer.plot_heatmap()