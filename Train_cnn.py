import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import random
import argparse
import numpy as np
import h5py
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

GEN4_W, GEN4_H = 1280, 720
NUM_CLASSES = 8 

def find_bbox_file(h5_path):
    base_dir = os.path.dirname(h5_path)
    base_name = os.path.basename(h5_path).replace('.h5', '')
    parts = base_name.split('_')
    timestamps = [p for p in parts if p.isdigit() and len(p) >= 8]
    if len(timestamps) >= 2:
        pattern = f"*{timestamps[0]}*{timestamps[1]}*bbox*.npy"
    else:
        pattern = f"*{'_'.join(parts[:3])}*bbox*.npy"
    cands = glob.glob(os.path.join(base_dir, pattern))
    return cands[0] if cands else None

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, h5_files, frames_per_file=10):
        self.samples = []
        for h5_path in h5_files:
            bbox_file = find_bbox_file(h5_path)
            if not bbox_file: continue
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'data' not in f: continue
                    T_len = f['data'].shape[0]
                    if T_len == 0: continue
                indices = np.linspace(0, T_len - 1, min(frames_per_file, T_len), dtype=int)
                for idx in indices:
                    self.samples.append((h5_path, bbox_file, idx))
            except Exception as e:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, bbox_file, frame_idx = self.samples[idx]
        with h5py.File(h5_path, 'r') as f:
            ds = f['data']
            frame = ds[frame_idx]
            delta_t = int(ds.attrs.get('delta_t', 50000))
            orig_w = int(ds.attrs.get('event_input_width', GEN4_W))
            orig_h = int(ds.attrs.get('event_input_height', GEN4_H))
            
        if frame.shape[0] < frame.shape[-1]:
            C = frame.shape[0]; H_img, W_img = frame.shape[1], frame.shape[2]
            neg = np.sum(frame[0::2], axis=0) if C > 1 else np.sum(frame, axis=0)
            pos = np.sum(frame[1::2], axis=0) if C > 1 else np.zeros_like(neg)
        else:
            H_img, W_img = frame.shape[0], frame.shape[1]; C = frame.shape[-1]
            neg = np.sum(frame[..., 0::2], axis=-1) if C > 1 else np.sum(frame, axis=-1)
            pos = np.sum(frame[..., 1::2], axis=-1) if C > 1 else np.zeros_like(neg)

        img = np.zeros((3, H_img, W_img), dtype=np.float32)
        img[0, :, :] = pos 
        img[2, :, :] = neg 
        
        max_val = max(pos.max(), neg.max())
        if max_val > 0: img = np.clip((img / max_val), 0, 1)

        t0 = frame_idx * delta_t
        t1 = t0 + delta_t
        bboxes_all = np.load(bbox_file)
        mask = (bboxes_all['t'] >= t0) & (bboxes_all['t'] <= t1)
        b_in_window = bboxes_all[mask]
        
        boxes, labels = [], []
        if len(b_in_window) > 0:
            best_t = np.max(b_in_window['t'])
            scale_x = W_img / float(orig_w); scale_y = H_img / float(orig_h)
            for b in b_in_window[b_in_window['t'] == best_t]:
                xmin = b['x'] * scale_x; ymin = b['y'] * scale_y
                xmax = xmin + (b['w'] * scale_x); ymax = ymin + (b['h'] * scale_y)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(b['class_id'] + 1)
                    
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes, "labels": labels,
            "image_id": torch.tensor([idx]), "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
        return torch.tensor(img, dtype=torch.float32), target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir', default=r"D:\Cours 2025-2026\P6 Projet event camera\test")
    p.add_argument('--epochs', type=int, default=30) 
    p.add_argument('--batch_size', type=int, default=2) 
    p.add_argument('--frames_per_file', type=int, default=10)
    args = p.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Appareil utilisé : {device} ")

    h5_files = sorted(glob.glob(os.path.join(args.test_dir, '**', '*.h5'), recursive=True))
    if not h5_files: return
        
    random.shuffle(h5_files)
    train_dataset = EventDataset(h5_files[:max(1, int(len(h5_files) * 0.8))], frames_per_file=args.frames_per_file)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES)
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)

    # --- PRÉPARATION SAUVEGARDE ---
    history_file = open("historique_loss.txt", "w")
    history_file.write("Epoch,Loss\n")
    all_losses = []

    print(f"\nDémarrage de l'entraînement Faster R-CNN pour {args.epochs} époques...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            # Anti-crash mémoire VRAM
            del images, targets, loss_dict, losses
            torch.cuda.empty_cache()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] - Batch [{i}/{len(train_loader)}]")

        avg_loss = epoch_loss / len(train_loader)
        all_losses.append(avg_loss)
        print(f"==> Fin Epoch {epoch+1} - Moyenne Loss: {avg_loss:.4f}\n")

        # Sauvegarde sur le disque dur instantanément
        history_file.write(f"{epoch+1},{avg_loss:.4f}\n")
        history_file.flush() # Force l'écriture physique sur le disque
        os.fsync(history_file.fileno())
        
        # Sauvegarde du modèle à chaque époque
        torch.save(model.state_dict(), "model_event_cam.pth")

    history_file.close()

    # --- GÉNÉRATION DU GRAPHIQUE ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), all_losses, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title(f"Courbe d'apprentissage - Faster R-CNN ({args.epochs} Époques)", fontsize=14, fontweight='bold')
    plt.xlabel("Époques", fontsize=12)
    plt.ylabel("Moyenne de l'erreur (Loss)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_ layout()
    plt.savefig("courbe_apprentissage_finale.png", dpi=300)
    print("✓ Entraînement terminé ! Courbe sauvegardée sous 'courbe_apprentissage_finale.png'")

if __name__ == '__main__':
    main()