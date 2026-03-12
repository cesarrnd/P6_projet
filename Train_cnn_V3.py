import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import random
import numpy as np
import h5py
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# --- PARAMÈTRES D'ENTRAÎNEMENT ---
EPOCHS = 50           # Objectif final pour un super score
BATCH_SIZE = 2        # Sécurité pour la VRAM de la GTX 1050
FRAMES_PER_FILE = 15  # Nombre d'images extraites par fichier h5
LEARNING_RATE = 0.002 # Taux d'apprentissage stable
TEST_DIR = r"D:\Cours 2025-2026\P6 Projet event camera\test"
MODEL_PATH = "model_event_cam.pth"

# ---------------------------------

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, h5_files, frames_per_file=15):
        self.samples = []
        for h5_path in h5_files:
            base_dir = os.path.dirname(h5_path)
            base_name = os.path.basename(h5_path).replace('.h5', '')
            parts = base_name.split('_')
            timestamps = [p for p in parts if p.isdigit() and len(p) >= 8]
            pattern = f"*{timestamps[0]}*{timestamps[1]}*bbox*.npy" if len(timestamps) >= 2 else f"*{'_'.join(parts[:3])}*bbox*.npy"
            cands = glob.glob(os.path.join(base_dir, pattern))
            bbox_file = cands[0] if cands else None
            if not bbox_file: continue
            try:
                with h5py.File(h5_path, 'r') as f:
                    T_len = f['data'].shape[0]
                    if T_len == 0: continue
                indices = np.linspace(0, T_len - 1, min(frames_per_file, T_len), dtype=int)
                for idx in indices: self.samples.append((h5_path, bbox_file, idx))
            except: pass

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        h5_path, bbox_file, frame_idx = self.samples[idx]
        with h5py.File(h5_path, 'r') as f:
            ds = f['data']; frame = ds[frame_idx]
            dt = int(ds.attrs.get('delta_t', 50000))
            ow, oh = int(ds.attrs.get('event_input_width', 1280)), int(ds.attrs.get('event_input_height', 720))
        
        # Reconstruction image (Polarité Pos/Neg)
        C, H, W = (frame.shape[0], frame.shape[1], frame.shape[2]) if frame.shape[0] < frame.shape[-1] else (frame.shape[-1], frame.shape[0], frame.shape[1])
        img_np = np.zeros((3, H, W), dtype=np.float32)
        if frame.shape[0] < frame.shape[-1]:
            img_np[0] = np.sum(frame[1::2], axis=0); img_np[2] = np.sum(frame[0::2], axis=0)
        else:
            img_np[0] = np.sum(frame[..., 1::2], axis=-1); img_np[2] = np.sum(frame[..., 0::2], axis=-1)
        
        mx = img_np.max()
        if mx > 0: img_np = np.clip(img_np / mx, 0, 1)

        # Bounding Boxes
        t0 = frame_idx * dt
        bboxes_all = np.load(bbox_file)
        mask = (bboxes_all['t'] >= t0) & (bboxes_all['t'] <= t0 + dt)
        b_win = bboxes_all[mask]
        boxes, labels = [], []
        if len(b_win) > 0:
            sx, sy = W/ow, H/oh
            for b in b_win[b_win['t'] == np.max(b_win['t'])]:
                boxes.append([b['x']*sx, b['y']*sy, (b['x']+b['w'])*sx, (b['y']+b['h'])*sy])
                labels.append(b['class_id']+1)

        target = {"boxes": torch.as_tensor(boxes if boxes else torch.zeros((0,4)), dtype=torch.float32),
                  "labels": torch.as_tensor(labels if labels else torch.zeros((0,), dtype=torch.int64), dtype=torch.int64),
                  "image_id": torch.tensor([idx]), "area": torch.zeros((len(labels),)), 
                  "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)}
        return torch.tensor(img_np), target

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Lancement de l'entraînement sur {device}")
    
    h5_files = glob.glob(os.path.join(TEST_DIR, '**', '*.h5'), recursive=True)
    dataset = EventDataset(h5_files, frames_per_file=FRAMES_PER_FILE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(8).to(device)
    
    # --- LA LIGNE MAGIQUE ANTI-AMNÉSIE ---
    if os.path.exists(MODEL_PATH):
        print(f"📥 Modèle trouvé ({MODEL_PATH}) ! Reprise de l'entraînement là où il s'était arrêté...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("🌱 Aucun modèle trouvé. Démarrage de zéro.")

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # "a" = append : On ajoute les lignes à la fin du fichier sans effacer l'historique
    history_log = open("historique_nuit.txt", "a") 
    losses_list = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad(); losses.backward(); optimizer.step()
            epoch_loss += losses.item()
            
            if i % 20 == 0: print(f"E{epoch+1} | B{i}/{len(loader)} | Loss: {losses.item():.4f}")
            
            # Vidage de la RAM vidéo pour éviter l'écran bleu
            del images, targets, loss_dict, losses
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(loader)
        losses_list.append(avg_loss)
        print(f"✅ EPOCH {epoch+1} TERMINEE - Moyenne: {avg_loss:.4f}")
        
        # Sauvegarde sécurisée sur le disque dur instantanément
        history_log.write(f"{epoch+1},{avg_loss:.4f}\n"); history_log.flush()
        torch.save(model.state_dict(), MODEL_PATH)

    history_log.close()
    
    # Génération du graphique de fin
    plt.plot(losses_list, marker='o', color='b')
    plt.title("Courbe d'apprentissage", fontweight='bold')
    plt.xlabel("Époques"); plt.ylabel("Loss")
    plt.grid(True, linestyle='--')
    plt.savefig("courbe_nuit.png")
    print("📊 Entraînement terminé. Courbe sauvegardée.")

if __name__ == "__main__": 
    main()