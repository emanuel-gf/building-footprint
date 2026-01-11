import cv2
import os
import numpy as np 

def get_classification_bbox(centroid, w, h, padding=0.1, aspect_ratio=None):
    """
    Constructs a bounding box based on a centroid and original dimensions.
    
    Args:
        centroid (tuple): (cX, cY) from moments.
        w (int): Original width of the contour.
        h (int): Original height of the contour.
        padding (float): Percentage to expand the box (e.g., 0.1 for 10%).
        aspect_ratio (float): Optional. Target width/height ratio (e.g., 1.0 for square).
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates.
    """
    cX, cY = centroid

    # 1. Apply Padding
    new_w = w * (1 + padding)
    new_h = h * (1 + padding)

    # 2. Enforce Aspect Ratio (if requested)
    if aspect_ratio is not None:
        current_ratio = new_w / new_h
        if current_ratio < aspect_ratio:
            # Too tall, expand width
            new_w = new_h * aspect_ratio
        else:
            # Too wide, expand height
            new_h = new_w / aspect_ratio

    # 3. Calculate Corners centered on (cX, cY)
    x1 = int(cX - (new_w / 2))
    y1 = int(cY - (new_h / 2))
    x2 = int(cX + (new_w / 2))
    y2 = int(cY + (new_h / 2))

    return x1, y1, x2, y2

def create_labels_yolov11_classification(
    list_bitmap, 
    output_folder, 
    skip_small_contour_areas=10, 
    padding=0.1, 
    aspect_ratio=1.0, 
    verbose=True
):
    """ 
    Generate YOLO-formatted label files from binary bitmap masks using geometric centroids.
    
    This function extracts the 'center of mass' (centroid) of polygons instead of 
    simple bounding box centers. It applies optional padding and enforces a specific 
    aspect ratio (useful for building classification) before normalizing coordinates.
    
    The output is a .txt file per image following the Ultralytics YOLO format:
    <class_id> <x_center> <y_center> <width> <height>
    
    Args:
        list_bitmap (list): Absolute paths to the bitmap mask images (binary 0-1 or 0-255).
        output_folder (str): Absolute path where .txt label files will be saved.
        skip_small_contour_areas (int): Minimum pixel area to consider a valid object.
        padding (float): Percentage (0.0 to 1.0) to expand the bounding box.
        aspect_ratio (float): Targeted width/height ratio. Set to 1.0 for square boxes.
        verbose (bool): If True, prints progress and warnings to console.
        
    Returns:
        list_contour_less_thre: List of tuples (path, index) for contours that 
                                fell below the area threshold.
    """
    os.makedirs(output_folder, exist_ok=True)
    if verbose:
        print(f"The files will be written at: {output_folder}")
    
    list_contour_less_thre = []
    
    for img_idx, img_path in enumerate(list_bitmap):
        # Prepare output path
        file_base = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_folder, f"{file_base}.txt")
        
        # Read image (OpenCV shape: [height, width])
        imgg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if imgg is None:
            continue
            
        img_h, img_w = imgg.shape
        
        # Ensure image is in 0-255 range for thresholding
        if imgg.max() <= 1.0:
            imgg = (imgg * 255).astype(np.uint8)

        _, thresh = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(output_path, 'w') as f:
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                
                if area < skip_small_contour_areas:
                    if verbose:
                        print(f"WARNING: Area ({area}) in {file_base} is below threshold.")
                    list_contour_less_thre.append((img_path, i))
                    continue
                
                # 1. Calculate Centroid (Center of Mass) via Moments
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                else:
                    # Fallback to bounding box center if moments fail
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cX, cY = bx + bw/2, by + bh/2

                # 2. Get original BBox dimensions
                _, _, w, h = cv2.boundingRect(cnt)

                # 3. Apply Padding
                new_w = w * (1 + padding)
                new_h = h * (1 + padding)

                # 4. Enforce Aspect Ratio
                if aspect_ratio is not None:
                    if (new_w / new_h) < aspect_ratio:
                        new_w = new_h * aspect_ratio
                    else:
                        new_h = new_w / aspect_ratio

                # 5. Normalize for YOLO (0.0 to 1.0)
                # x_center, y_center, width, height
                norm_xc = cX / img_w
                norm_yc = cY / img_h
                norm_w = new_w / img_w
                norm_h = new_h / img_h

                # Clamp values to [0.0, 1.0] to prevent YOLO errors
                norm_xc = max(0, min(1, norm_xc))
                norm_yc = max(0, min(1, norm_yc))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                f.write(f"0 {norm_xc:.8f} {norm_yc:.8f} {norm_w:.8f} {norm_h:.8f}\n")

        if verbose:
            progress = (img_idx + 1) / len(list_bitmap)
            print(f"Processed: {file_base}.txt | {progress:.2%}")

    return list_contour_less_thre







def create_labels_yolov11_segmentation(list_bitmap, output_folder, skip_small_contour_areas=10,percent_epsilon=0.001, verbose=True):
    """ 
    Create a label.txt file, saving from a list of file names, in the specifiied folder, where the output_file is the same name as the input bitmap file.
    
    The labels follow the guideline of Ultralytics, where Each line holds an object with `class x_center y_center width height`
    Box coordinates must be in normalized xywh format (from 0 to 1). 
    This is designed for only ONE class number. It does not designed for a multi-class img_bitmap. 
    
    -----
    args:
        list_bitmap:
        A list of image paths. The path it is the absolute path of the image.
            A mask image range from (0,1).
        output_folder:
            Absolut path of the folder where to save it       
        skip_small_contour_areas:int 
            Sometimes the contour size of an given set of binary colors is small, by threholding this small contour helps to only delineate the correct binary ones. The area is in pixel size.
            This represent the percentual of the perimeter. A percent_episolon of 0.01 will return a epsilon of= perimter*percent_epsilon, which is 1% of the perimeter.
            Default=0.001 = 0.01%
    return:
        list_contour_less_thre:
            absolut path of partially/total unsuceed paths and the position of the contour
    """
    ## Create the dirs if does not exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"The files are be written at: {output_folder}")
    
    list_contour_less_thre = []
    num_verbose=1
    ## Iter through the list
    for img_path in list_bitmap:
        ## Create the output_path
        name_file = ".".join([os.path.basename(img_path).split('.')[0],'txt'])
        output_path = os.path.join(output_folder, name_file)
        
        ## read the image
        imgg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        ## need to rescale the image to 0-255 instead of 0-1 
        imgg_ = imgg*255

        ## get vars
        imgg_width, imgg_height = imgg.shape

        ## threshold
        _, thresh = cv2.threshold(imgg_, 127, 255, 0)

        ## contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Open output file
        with open(output_path, 'w') as f:
            
            # Process each contour
            for i,cnt in enumerate(contours):
                # Skip very small contours (noise)
                if cv2.contourArea(cnt) < 10:
                    print(f"WARNING: Area of contour is lesser than threshold!")
                    list_contour_less_thre.append((img_path, i))
                
                ## Higher the epsilon, higher are the threshold of the Douglas-Peucked algorithm. Which means far and more rectangular than a smaller episolon
                epsilon = cv2.arcLength(cnt, True)*percent_epsilon ## 0.1 of the arclength (perimeter)
                
                
                ## Smooth the mask in case of very complexed geometries. 
                approx =  cv2.approxPolyDP(cnt, epsilon, True)
                
                # Start with class_id
                line = "0"
                
                # Add normalized coordinates
                for point in approx:
                    x = point[0][0] / imgg_width
                    y = point[0][1] / imgg_height
                    line += f" {x:.8f} {y:.8f}"
                
                f.write(line + "\n")

        
        if verbose==True:
            print(f"Processed: {name_file} | {num_verbose/len(list_bitmap):.2%}")
            num_verbose+=1
    return list_contour_less_thre


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


def read_classication_label(txt_path):
    """
    Read a YOLO-v.11 object detection label file (bounding boxes).
    
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