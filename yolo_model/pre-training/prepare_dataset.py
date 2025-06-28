import os
import shutil
import random
from pathlib import Path

def prepare_pharma_dataset():
    """
    Organize pharmaceutical dataset for YOLOv8 classification
    """
    # Source dataset path
    source_path = "/home/ubuntu/.cache/kagglehub/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images/versions/4/Drug Vision/Data Combined"
    
    # Target dataset path
    target_path = "/home/ubuntu/pharma_system/yolo_model/dataset"
    
    # Drug classes
    drug_classes = [
        'Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc',
        'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep'
    ]
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        for drug_class in drug_classes:
            target_dir = os.path.join(target_path, split, drug_class)
            os.makedirs(target_dir, exist_ok=True)
    
    print(f"Created dataset structure at: {target_path}")
    
    # Split ratios: 70% train, 15% val, 15% test
    total_images = 0
    
    for drug_class in drug_classes:
        source_class_dir = os.path.join(source_path, drug_class)
        
        # Get all images in the class directory
        images = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)  # Shuffle for random split
        
        total_count = len(images)
        train_count = int(0.7 * total_count)
        val_count = int(0.15 * total_count)
        test_count = total_count - train_count - val_count
        
        print(f"{drug_class}: {total_count} images -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        # Split and copy files
        splits = {
            'train': images[:train_count],
            'val': images[train_count:train_count + val_count],
            'test': images[train_count + val_count:]
        }
        
        for split_name, split_images in splits.items():
            target_split_dir = os.path.join(target_path, split_name, drug_class)
            
            for image in split_images:
                source_file = os.path.join(source_class_dir, image)
                target_file = os.path.join(target_split_dir, image)
                shutil.copy2(source_file, target_file)
        
        total_images += total_count
    
    print(f"\nDataset preparation completed!")
    print(f"Total images processed: {total_images}")
    print(f"Dataset location: {target_path}")
    
    # Create data.yaml for YOLOv8
    yaml_content = f"""# Pharmaceutical Drug Classification Dataset
path: {target_path}
train: train
val: val
test: test

# Number of classes
nc: {len(drug_classes)}

# Class names
names:
"""
    for i, drug_class in enumerate(drug_classes):
        yaml_content += f"  {i}: {drug_class}\n"
    
    yaml_path = os.path.join(target_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml at: {yaml_path}")
    
    return target_path

if __name__ == "__main__":
    random.seed(42)  # For reproducible splits
    dataset_path = prepare_pharma_dataset() 