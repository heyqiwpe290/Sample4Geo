import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from dataset.vigor import VigorDatasetEval
from transforms import get_transforms_val
from evaluate.vigor import evaluate
from model import TimmModel


@dataclass
class EvaluationConfig:
    """Configuration settings for VIGOR same-area dataset evaluation"""
    
    # Model Configuration
    model_name: str = 'convnext_base.fb_in22k_ft_in1k_384'
    image_size: int = 384
    
    # Evaluation Parameters
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    
    # Dataset Configuration
    data_folder: str = "./data/VIGOR"
    same_area: bool = True  # Set to True for same-area evaluation
    ground_cutting: int = 0  # Ground image cutting pixels
    
    # Model Checkpoint
    checkpoint_start: str = 'pretrained/vigor_same/convnext_base.fb_in22k_ft_in1k_384/weights_e40_0.7786.pth'
    
    # System Settings
    num_workers: int = 0 if os.name == 'nt' else 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Initialize configuration
    config = EvaluationConfig()
    
    # Initialize model
    print(f"\nModel: {config.model_name}")
    model = TimmModel(
        config.model_name,
        pretrained=True,
        img_size=config.image_size
    )
    
    # Get model configuration
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.image_size
    
    # Calculate image sizes
    image_size_sat = (img_size, img_size)
    new_width = img_size * 2
    new_height = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    img_size_ground = (new_height, new_width)
    
    # Load pretrained checkpoint
    if config.checkpoint_start:
        print(f"Start from: {config.checkpoint_start}")
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)
    
    # Configure multi-GPU if available
    print(f"GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    
    # Move model to device
    model = model.to(config.device)
    
    print(f"\nImage Size Sat: {image_size_sat}")
    print(f"Image Size Ground: {img_size_ground}")
    print(f"Mean: {mean}")
    print(f"Std: {std}\n")
    
    # Initialize validation transforms
    sat_transforms_val, ground_transforms_val = get_transforms_val(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
        ground_cutting=config.ground_cutting
    )
    
    # Initialize test datasets
    reference_dataset_test = VigorDatasetEval(
        data_folder=config.data_folder,
        split="test",
        img_type="reference",
        same_area=config.same_area,
        transforms=sat_transforms_val
    )
    
    reference_dataloader_test = DataLoader(
        reference_dataset_test,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    # Initialize query dataset
    query_dataset_test = VigorDatasetEval(
        data_folder=config.data_folder,
        split="test",
        img_type="query",
        same_area=config.same_area,
        transforms=ground_transforms_val
    )
    
    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"Query Images Test: {len(query_dataset_test)}")
    print(f"Reference Images Test: {len(reference_dataset_test)}")
    
    # Run evaluation
    print(f"\n{'-'*30}[VIGOR Same]{'-'*30}")
    
    r1_test = evaluate(
        config=config,
        model=model,
        reference_dataloader=reference_dataloader_test,
        query_dataloader=query_dataloader_test,
        ranks=[1, 5, 10],
        step_size=1000,
        cleanup=True
    )
    
    return r1_test


if __name__ == '__main__':
    main()