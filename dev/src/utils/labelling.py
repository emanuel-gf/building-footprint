import cv2
import os

def create_labels_yolov11_classification(list_bitmap, output_folder, skip_small_contour_areas=10, verbose=True):
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
            Sometimes the contour size is small and threholding this small contours help to only delineate the correct ones. The are is in pixel size.
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
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                
                # (normalized coordinates)  
                x_center, y_center, w_, h_ = (x + w/2) / imgg_width, (y + h/2) / imgg_height, w / imgg_width, h / imgg_height

                # Write to file: class_id x_center y_center width height
                f.write(f"0 {x_center:.8f} {y_center:.8f} {w_:.8f} {h_:.8f}\n")

        
        if verbose==True:
            print(f"Processed: {name_file} | {num_verbose/len(list_bitmap):.2%}")
            num_verbose+=1
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
            Sometimes the contour size is small and threholding this small contours help to only delineate the correct ones. The are is in pixel size.
        percent_epsilon:float
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