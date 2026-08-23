import os
import time
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np

from baseline_model import BestenSingle
from dataset import UnifiedPedestrianDataset
from utils import MaskedBCEWithLogitsLoss, calculate_masked_metrics, rand_bbox

# --- UPDATED 21-ATTRIBUTE SCHEMA ---
SCHEMA_KEYS = [
    "Female", "Male", "Age_Child", "Age_Adult", "Bald", "Short_Hair", "Long_Hair",
    "Backpack", "Hat", "Glasses", "Handbag", "MessengerBag", "PlasticBag",
    "ShortSleeve", "LongSleeve", "Trousers", "Shorts", "Skirt_or_Dress",
    "Boots", "Sneakers", "LeatherShoes"
]

def get_train_transforms():
    return T.Compose([
    # 1. Spatial & Color Augmentations (Applied to the PIL Image)
        T.RandomHorizontalFlip(p=0.5), 
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        
        # 2. Convert to Tensor
        T.ToTensor(),
        
        # 3. Normalize (Must happen before RandomErasing)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 4. Occlusion (Applied to the Tensor)
        T.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

def get_val_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_model(run_name="baseline", start_lr=0.1, end_lr=1e-5, total_epochs=20, p_mix_schedule=None, resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate default schedule if none provided
    if p_mix_schedule is None:
        p_mix_schedule = [0.0] * total_epochs

    train_dataset = UnifiedPedestrianDataset("./data/unified_train.csv", transform=get_train_transforms())
    val_dataset = UnifiedPedestrianDataset("./data/unified_val.csv", transform=get_val_transforms())
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True , persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    current_layer_config = [(4, 64, 2), (4, 128, 2), (4, 256, 2), (4, 512, 2), (1, 1024, 2)]

    model = BestenSingle(
        num_classes=len(SCHEMA_KEYS), 
        layer_config=current_layer_config
    ).to(device)

    criterion = MaskedBCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=end_lr)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_f1 = 0.0
    
    # --- RESUME LOGIC ---
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        total_epochs = checkpoint['total_epochs']
        p_mix_schedule = checkpoint['p_mix_schedule']
        print(f"Resuming from Epoch {start_epoch} | Last LR: {checkpoint['last_lr']:.5f}")

    log_dir = Path(f"logs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}_log.json"

    print("="*85)
    print(f"Starting {run_name} | Epochs: {total_epochs} | LR Range: {start_lr} -> {end_lr}")
    print("="*85)

    for epoch in range(start_epoch, total_epochs + 1):
        start_time = time.time()
        lr = scheduler.get_last_lr()[0]
        
        # Safely grab this epoch's p_mix from the schedule
        current_p_mix = p_mix_schedule[epoch - 1]

        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{total_epochs} [Train]", leave=False)
        
        for images, labels, masks in train_pbar:
            images, labels, masks = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Dynamic CutMix Application
            r = np.random.rand()
            if r < current_p_mix:
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(images.size()[0]).cuda()
                
                target_a, target_b = labels, labels[rand_index]
                mask_a, mask_b = masks, masks[rand_index]
                
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                
                with torch.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, target_a, mask_a) * lam + criterion(outputs, target_b, mask_b) * (1. - lam)
            else:
                with torch.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # --- VAL ---
        model.eval()
        val_loss = 0.0
        all_logits, all_targets, all_masks = [], [], []
        val_pbar = tqdm(val_loader, desc=f"Ep {epoch:02d}/{total_epochs} [Val]  ", leave=False)
        
        with torch.no_grad():
            for images, labels, masks in val_pbar:
                images, labels, masks = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels, masks)
                
                val_loss += loss.item()
                all_logits.append(outputs.cpu())
                all_targets.append(labels.cpu())
                all_masks.append(masks.cpu())

        scheduler.step()
        
        # --- METRICS & SAVING ---
        epoch_duration = time.time() - start_time
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        
        full_logits = torch.cat(all_logits, dim=0)
        full_targets = torch.cat(all_targets, dim=0)
        full_masks = torch.cat(all_masks, dim=0)
        val_metrics = calculate_masked_metrics(full_logits, full_targets, full_masks)
        
        v_prec, v_rec, v_f1 = val_metrics['precision'] * 100, val_metrics['recall'] * 100, val_metrics['f1'] * 100
        
        # TUI Printout now shows the active p_mix
        print(f"Ep {epoch:2d}/{total_epochs} | Time: {epoch_duration:.1f}s | LR: {lr:.5f} | p_mix: {current_p_mix}")
        print(f"Loss -> Train: {t_loss:.3f} | Val: {v_loss:.3f} | Metrics -> F1: {v_f1:.1f}%")
        
        # Save Stateful Checkpoint with New Metadata
        if v_f1 > best_f1:
            best_f1 = v_f1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_f1': best_f1,
                'schema': SCHEMA_KEYS,
                'layer_config': current_layer_config, # <--- New identifier!
                'start_lr': start_lr,
                'end_lr': end_lr,
                'last_lr': lr, 
                'total_epochs': total_epochs,
                'p_mix_schedule': p_mix_schedule
            }
            torch.save(checkpoint, checkpoint_dir / f"{run_name}_best.pth")
            print("--> Saved new best stateful checkpoint!\n")
        else:
            print("\n")