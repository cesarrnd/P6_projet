"""
train_simple_cnn.py — Entraînement direct d'un CNN (Faster R-CNN) sur les Tenseurs Prophesee.
Aucune création de fichier supplémentaire requise, tout est fait en mémoire !
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import random
import argparse
import numpy as np
import h5py
import torch
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Dimensions d'origine (utilisées pour remettre à l'échelle les BBoxes si besoin)
GEN4_W, GEN4_H = 1280, 720

# 7 Classes + 1 (le fond/background, obligatoire pour PyTorch)
NUM_CLASSES = 7 + 1 

def find_bbox_file(h5_path):
    """Trouve le fichier de bounding boxes correspondant."""
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
            if not bbox_file:
                continue
                
            try:
                # Vérifier la taille du dataset H5
                with h5py.File(h5_path, 'r') as f:
                    if 'data' not in f: continue
                    T_len = f['data'].shape[0]
                    if T_len == 0: continue
                    
                # On stocke le chemin et les indices des frames à utiliser
                indices = np.linspace(0, T_len - 1, min(frames_per_file, T_len), dtype=int)
                for idx in indices:
                    self.samples.append((h5_path, bbox_file, idx))
            except Exception as e:
                print(f"Erreur avec {h5_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, bbox_file, frame_idx = self.samples[idx]
        
        # 1. Charger la frame H5
        with h5py.File(h5_path, 'r') as f:
            ds = f['data']
            frame = ds[frame_idx]
            delta_t = int(ds.attrs.get('delta_t', 50000))
            orig_w = int(ds.attrs.get('event_input_width', GEN4_W))
            orig_h = int(ds.attrs.get('event_input_height', GEN4_H))
            
        # Tenseur vers RGB (Bleu = Neg, Rouge = Pos)
        if frame.shape[0] < frame.shape[-1]:
            C = frame.shape[0]; H_img, W_img = frame.shape[1], frame.shape[2]
            neg = np.sum(frame[0::2], axis=0) if C > 1 else np.sum(frame, axis=0)
            pos = np.sum(frame[1::2], axis=0) if C > 1 else np.zeros_like(neg)
        else:
            H_img, W_img = frame.shape[0], frame.shape[1]; C = frame.shape[-1]
            neg = np.sum(frame[..., 0::2], axis=-1) if C > 1 else np.sum(frame, axis=-1)
            pos = np.sum(frame[..., 1::2], axis=-1) if C > 1 else np.zeros_like(neg)

        img = np.zeros((3, H_img, W_img), dtype=np.float32)
        img[0, :, :] = pos  # Canal Rouge
        img[2, :, :] = neg  # Canal Bleu
        
        # Normalisation
        max_val = max(pos.max(), neg.max())
        if max_val > 0:
            img = np.clip((img / max_val), 0, 1)

        # 2. Charger les Bounding Boxes
        t0 = frame_idx * delta_t
        t1 = t0 + delta_t
        bboxes_all = np.load(bbox_file)
        mask = (bboxes_all['t'] >= t0) & (bboxes_all['t'] <= t1)
        b_in_window = bboxes_all[mask]
        
        boxes = []
        labels = []
        
        if len(b_in_window) > 0:
            best_t = np.max(b_in_window['t'])
            b_final = b_in_window[b_in_window['t'] == best_t]
            
            scale_x = W_img / float(orig_w)
            scale_y = H_img / float(orig_h)
            
            for b in b_final:
                # Coordonnées (xmin, ymin, xmax, ymax)
                xmin = b['x'] * scale_x
                ymin = b['y'] * scale_y
                xmax = xmin + (b['w'] * scale_x)
                ymax = ymin + (b['h'] * scale_y)
                
                # Sécurité PyTorch (les boîtes valides doivent avoir l'aire > 0)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    # PyTorch demande que la classe 0 soit le background, on décale donc de +1
                    labels.append(b['class_id'] + 1)
                    
        # Gestion des images vides (sans objet)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        return torch.tensor(img, dtype=torch.float32), target

def collate_fn(batch):
    """Fonction utilitaire pour regrouper les données d'une batch."""
    return tuple(zip(*batch))

def get_model(num_classes):
    """Charge un Faster R-CNN pré-entraîné et adapte la tête de classification."""
    # Modèle pré-entraîné sur des images classiques (COCO)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # On remplace la dernière couche pour s'adapter à nos 8 classes (7 + background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir', default=r"D:\Cours 2025-2026\P6 Projet event camera\test")
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--frames_per_file', type=int, default=10, help="Frames à piocher par fichier")
    args = p.parse_args()

    # Configuration de l'appareil (GPU si disponible, sinon CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Appareil utilisé : {device}")

    # 1. Recherche et séparation des fichiers (80% Train, 20% Val)
    h5_files = sorted(glob.glob(os.path.join(args.test_dir, '**', '*.h5'), recursive=True))
    if not h5_files:
        print(f"Aucun .h5 trouvé dans {args.test_dir}")
        return
        
    random.shuffle(h5_files)
    split_idx = max(1, int(len(h5_files) * 0.8))
    train_files = h5_files[:split_idx]
    val_files = h5_files[split_idx:]
    
    print(f"Fichiers: {len(train_files)} Entraînement / {len(val_files)} Validation")

    # 2. Création des Datasets
    train_dataset = EventDataset(train_files, frames_per_file=args.frames_per_file)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    print(f"Images d'entraînement générées : {len(train_dataset)}")

    # 3. Initialisation du modèle
    model = get_model(NUM_CLASSES)
    model.to(device)

    # Optimiseur classique
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 4. Boucle d'entraînement
    print("\nDémarrage de l'entraînement...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # PyTorch gère le calcul de la loss tout seul !
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if i % 5 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] - Batch [{i}/{len(train_loader)}] - Loss: {losses.item():.4f}")

        print(f"==> Fin Epoch {epoch+1} - Moyenne Loss: {epoch_loss/len(train_loader):.4f}\n")

    # 5. Sauvegarde
    torch.save(model.state_dict(), "model_event_cam.pth")
    print("✓ Entraînement terminé ! Modèle sauvegardé sous 'model_event_cam.pth'")

if __name__ == '__main__':
    main()