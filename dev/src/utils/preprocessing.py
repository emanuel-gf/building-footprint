from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split 


def create_yolo_tree_structure(list_imgs,
                               name_folder_label,
                               output_dir,
                               random_state=42, 
                               test_size=0.1):
    """
    Create a folder that follows the guidelines of folder structure for YOLO models. 
    Split between train and test. Does not account for validation dataset.
    
    args:
        list_imgs: List with image paths (Path objects or strings)
        name_folder_label: Name of the folder where the labels are stored
        output_dir: The output directory where to put the images
        random_state: Random seed for reproducibility
        test_size: Proportion of images to use for testing (0.0 to 1.0)
    
    returns:
        dict: Summary with counts for each subset
    """
    # Ensure list_imgs contains Path objects
    list_imgs = [Path(img) if not isinstance(img, Path) else img for img in list_imgs]
    
    # Split the dataset
    train_, test_ = train_test_split(list_imgs, test_size=test_size, random_state=random_state)
    
    output_dir = Path(output_dir)
    summary = {}
    
    print(f"\n{'='*60}")
    print(f"Creating YOLO dataset structure in: {output_dir}")
    print(f"{'='*60}")
    print(f"Total images: {len(list_imgs)}")
    print(f"Train/Test split: {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")
    print(f"{'='*60}\n")
    
    ## Create the directory in case does not exist 
    os.makedirs(output_dir, exist_ok=True)
    
    for subset, subset_images in [('train', train_), ('test', test_)]:
        images_dir = output_dir / "images" / subset
        labels_dir = output_dir / "labels" / subset
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Counters
        copied_images = 0
        copied_labels = 0
        missing_labels = []
        
        # Copy files
        for image in subset_images:
            # Copy image
            dest_image = images_dir / image.name
            shutil.copy(image, dest_image)
            copied_images += 1
            
            # Copy corresponding label
            label_file = image.stem + '.txt'
            source_label = image.parent.parent / name_folder_label / label_file
            
            if source_label.exists():
                dest_label = labels_dir / label_file
                shutil.copy(source_label, dest_label)
                copied_labels += 1
            else:
                missing_labels.append(image.name)
        
        # Store summary
        summary[subset] = {
            'images': copied_images,
            'labels': copied_labels,
            'missing_labels': len(missing_labels)
        }
        
        # Print results for this subset
        print(f"  {subset.upper()} set:")
        print(f"   Images copied: {copied_images}")
        print(f"   Labels copied: {copied_labels}")
        
        if missing_labels:
            print(f" Missing labels: {len(missing_labels)}")
            if len(missing_labels) <= 5:
                for img in missing_labels:
                    print(f"      - {img}")
            else:
                print(f"      - {missing_labels[0]}")
                print(f"      - ... and {len(missing_labels) - 1} more")

    
    return summary