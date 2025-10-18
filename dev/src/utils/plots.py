import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.patches as patches
from labelling import read_classication_label, read_segmentation_label

def print_img(path, dir="/mnt/d/desktop/drone-mapping/code/dev/newdataset_overlap/patches"):
        
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert BGR to RGB
    plt.imshow(img)
    plt.show()
    


def plot_images(image_paths,n=None, cols=5, figsize=(15, 5), titles=None):
    """
    Plots n images from a list of image paths using matplotlib.

    Parameters:
        image_paths (list): List of image file paths.
        n (int): Number of images to display. Default is 5.
        cols (int): Number of columns in the grid. Default is 5.
        figsize (tuple): Size of the entire figure. Default is (15, 5).
        titles (list): Optional list of titles for each image.

    Returns:
        None
    """
    if n is None:
        n =  len(image_paths)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(cols * rows):
        ax = axes[i]
        if i < n:
            img_path = image_paths[i]
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax.imshow(img)
                if titles and i < len(titles):
                    ax.set_title(titles[i])
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
                ax.set_facecolor('lightgray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_pairwise(pairwise_list_file,
                  n=None, cols=4,
                  figsize=(15, 5),
                  titles=None):
    """
    Plots n pairs of images from a list of image path pairs using matplotlib.
    
    The order of the pair tuple is (patch, bitmap)

    Parameters:
        pairwise_list_file (list): List of tuples, where each tuple is (patch_path, bitmap_path).
        n (int): Number of pairs to display. Default is all pairs.
        cols (int): Number of columns in the grid. Default is 4.
        figsize (tuple): Size of the entire figure. Default is (15, 5).
        titles (list): Optional list of titles for each pair.

    Returns:
        None
    """
    
    # Determine number of pairs to plot
    if n is None:
        n = len(pairwise_list_file)
    else:
        n = min(n, len(pairwise_list_file))  # Don't exceed available pairs
    
    if n == 0:
        print("No image pairs to plot.")
        return
    
    # Force cols to 4 (2 pairs per row, since each pair takes 2 columns)
    cols = 4  
    pairs_per_row = cols // 2  # Each pair needs 2 columns
    rows = (n + pairs_per_row - 1) // pairs_per_row  # Calculate rows needed
    
    print(f"Plotting {n} pairs in {rows} rows x {pairs_per_row} pairs per row")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle the case where we have only one row
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [[axes]]
    
    # Flatten for easier indexing, but keep track of position
    axes_flat = axes.flatten() if n > 1 or cols > 1 else [axes]

    plot_index = 0  # Track which subplot we're on
    
    for pair_idx in range(n):
        if pair_idx < len(pairwise_list_file):
            img_path_patch, img_path_bitmap = pairwise_list_file[pair_idx]
            
            # Check if both images exist
            patch_exists = os.path.exists(img_path_patch)
            bitmap_exists = os.path.exists(img_path_bitmap)
            
            # Plot patch image
            if plot_index < len(axes_flat):
                if patch_exists:
                    try:
                        img = plt.imread(img_path_patch)
                        axes_flat[plot_index].imshow(img)
                        if titles and pair_idx < len(titles):
                            axes_flat[plot_index].set_title(f"Patch: {titles[pair_idx]}")
                        else:
                            axes_flat[plot_index].set_title(f"Patch: {os.path.basename(img_path_patch).split('.')[0]}")
                    except Exception as e:
                        axes_flat[plot_index].text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path_patch)}", 
                                                 ha='center', va='center', fontsize=8)
                        axes_flat[plot_index].set_facecolor('lightcoral')
                else:
                    axes_flat[plot_index].text(0.5, 0.5, f"Patch not found\n{os.path.basename(img_path_patch)}", 
                                             ha='center', va='center', fontsize=8)
                    axes_flat[plot_index].set_facecolor('lightgray')
                
                axes_flat[plot_index].axis('off')
                plot_index += 1
            
            # Plot bitmap image
            if plot_index < len(axes_flat):
                if bitmap_exists:
                    try:
                        img_bitmap = plt.imread(img_path_bitmap)
                        # Handle different bitmap formats
                        if img_bitmap.dtype == bool:
                            img_bitmap = img_bitmap.astype(float)
                        axes_flat[plot_index].imshow(img_bitmap)
                        if titles and pair_idx < len(titles):
                            axes_flat[plot_index].set_title(f"Bitmap: {titles[pair_idx]}")
                        else:
                            axes_flat[plot_index].set_title(f"Bitmap: {os.path.basename(img_path_bitmap).split('.')[0]}")
                    except Exception as e:
                        axes_flat[plot_index].text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path_bitmap)}", 
                                                 ha='center', va='center', fontsize=8)
                        axes_flat[plot_index].set_facecolor('lightcoral')
                else:
                    axes_flat[plot_index].text(0.5, 0.5, f"Bitmap not found\n{os.path.basename(img_path_bitmap)}", 
                                             ha='center', va='center', fontsize=8)
                    axes_flat[plot_index].set_facecolor('lightgray')
                
                axes_flat[plot_index].axis('off')
                plot_index += 1

    # Hide any remaining empty subplots
    for i in range(plot_index, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_pairwise_simple(pairwise_list_file, n=None, figsize=(15, 10)):
    """
    Simplified version that plots pairs in a 2-column layout (patch | bitmap)
    
    Parameters:
        pairwise_list_file (list): List of tuples (patch_path, bitmap_path)
        n (int): Number of pairs to display
        figsize (tuple): Figure size
    """
    
    if n is None:
        n = len(pairwise_list_file)
    else:
        n = min(n, len(pairwise_list_file))
    
    if n == 0:
        print("No pairs to plot")
        return
    
    if n> 12:
        print("Too many pairs to plot at once.")
        n=12

    
    fig, axes = plt.subplots(n, 2, figsize=figsize)
    
    # Handle single pair case
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        if i < len(pairwise_list_file):
            patch_path, bitmap_path = pairwise_list_file[i]
            
            # Plot patch
            if os.path.exists(patch_path):
                try:
                    patch_img = plt.imread(patch_path)
                    axes[i, 0].imshow(patch_img)
                    axes[i, 0].set_title(f"Patch: {os.path.basename(patch_path)}")
                except:
                    axes[i, 0].text(0.5, 0.5, "Error loading patch", ha='center', va='center')
                    axes[i, 0].set_facecolor('lightcoral')
            else:
                axes[i, 0].text(0.5, 0.5, "Patch not found", ha='center', va='center')
                axes[i, 0].set_facecolor('lightgray')
            
            axes[i, 0].axis('off')
            
            # Plot bitmap
            if os.path.exists(bitmap_path):
                try:
                    bitmap_img = plt.imread(bitmap_path)
                    axes[i, 1].imshow(bitmap_img)
                    axes[i, 1].set_title(f"Bitmap: {os.path.basename(bitmap_path)}")
                except:
                    axes[i, 1].text(0.5, 0.5, "Error loading bitmap", ha='center', va='center')
                    axes[i, 1].set_facecolor('lightcoral')
            else:
                axes[i, 1].text(0.5, 0.5, "Bitmap not found", ha='center', va='center')
                axes[i, 1].set_facecolor('lightgray')
            
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    

def read_segmentation_label(txt_path):
    """
    Read a YOLO segmentation label file.
    
    Args:
        txt_path: Path to the .txt label file
        
    Returns:
        list of dictionaries, each containing:
            - 'class_id': int
            - 'points': numpy array of shape (N, 2) with normalized coordinates
    """
    objects = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:  # Need at least class_id + one point (x, y)
                continue
            
            class_id = int(parts[0])
            
            # Extract coordinate pairs
            coords = [float(x) for x in parts[1:]]
            points = np.array(coords).reshape(-1, 2)  # Reshape to (N, 2)
            
            objects.append({
                'class_id': class_id,
                'points': points
            })
    
    return objects

def plot_segmentation_label(txt_path, image_path=None, figsize=(10, 10)):
    """
    Plot YOLO segmentation labels, optionally overlaid on an image.
    
    Args:
        txt_path: Path to the .txt label file
        image_path: Optional path to the corresponding image
        figsize: Figure size (width, height)
    """
    if image_path:
        filename = os.path.basename(image_path)
    else:
        filename = os.path.basename(txt_path)
    objects = read_segmentation_label(txt_path)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # If image provided, display it
    if image_path:
        import cv2
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        img_height, img_width = img.shape[:2]
    else:
        # Just plot on normalized coordinates
        img_width, img_height = 1, 1
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip y-axis to match image coordinates
        ax.set_aspect('equal')
    
    # Plot each object
    colors = plt.cm.rainbow(np.linspace(0, 1, len(objects)))
    
    for i, obj in enumerate(objects):
        points = obj['points'].copy()
        
        # Convert normalized coordinates to pixel coordinates
        points[:, 0] *= img_width
        points[:, 1] *= img_height
        
        # Close the polygon by adding first point at the end
        points_closed = np.vstack([points, points[0]])
        
        # Plot the polygon
        ax.plot(points_closed[:, 0], points_closed[:, 1], 
                color=colors[i], linewidth=2, marker='o', markersize=4,
                label=f'Object {i+1} (class {obj["class_id"]})')
        
        # Fill the polygon with transparency
        ax.fill(points_closed[:, 0], points_closed[:, 1], 
                color=colors[i], alpha=0.3)
    
    ax.set_title(f"{filename}")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

def read_classication_label(txt_path):
    """
    Read a YOLO object detection label file (bounding boxes).
    
    Format: class_id x_center y_center width height
    All coordinates are normalized (0-1).
    
    Args:
        txt_path: Path to the .txt label file
        
    Returns:
        list of dictionaries, each containing:
            - 'class_id': int
            - 'x_center': float (normalized)
            - 'y_center': float (normalized)
            - 'width': float (normalized)
            - 'height': float (normalized)
    """
    boxes = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:  # Must have exactly 5 values
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            boxes.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    return boxes


def plot_classification_label(txt_path, image_path=None, figsize=(10, 10), 
                        class_names=None, show_labels=True):
    """
    Plot YOLO detection bounding boxes, optionally overlaid on an image.
    
    Args:
        txt_path: Path to the .txt label file
        image_path: Optional path to the corresponding image
        figsize: Figure size (width, height)
        class_names: Optional dictionary mapping class_id to class name
        show_labels: Whether to show box labels with class and coordinates
    """
    boxes = read_classication_label(txt_path)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # If image provided, display it
    if image_path:
        import cv2
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        img_height, img_width = img.shape[:2]
        filename = os.path.basename(image_path)
    else:
        # Just plot on normalized coordinates
        img_width, img_height = 1, 1
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip y-axis to match image coordinates
        ax.set_aspect('equal')
        filename = os.path.basename(filename)
    
    # Plot each bounding box
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(boxes), 1)))
    
    for i, box in enumerate(boxes):
        # Convert from center format to corner format
        x_center = box['x_center'] * img_width
        y_center = box['y_center'] * img_height
        width = box['width'] * img_width
        height = box['height'] * img_height
        
        # Calculate top-left corner
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=colors[i], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add center point
        ax.plot(x_center, y_center, 'o', color=colors[i], markersize=6)
        
        # Add label
        if show_labels:
            class_id = box['class_id']
            if class_names and class_id in class_names:
                label = f"{class_names[class_id]}"
            else:
                label = f"Class {class_id}"
            
            # Position label at top-left corner of box
            ax.text(x_min, y_min - 5, label,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7),
                   fontsize=10, color='white', weight='bold')
    
    ax.set_title(f'{filename}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()