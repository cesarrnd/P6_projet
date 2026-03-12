import os
import glob
import numpy as np
import h5py
import cv2
import torch
from ultralytics import YOLO

# ==========================================
# PARAMÈTRES DU PROJET P6
# ==========================================
TEST_DIR = r"D:\Cours 2025-2026\P6 Projet event camera\test"
OUTPUT_DIR = "dataset_yolo"
YAML_FILE = "dataset.yaml"

GEN4_W, GEN4_H = 1280, 720
FRAMES_PER_FILE = 15
EPOCHS = 30
BATCH_SIZE = 4
# ==========================================

# La même fonction infaillible que dans ton CNN Robuste
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

def phase1_prepare_data():
    print("\n" + "="*40)
    print("🚀 PHASE 1 : Préparation des données YOLO")
    print("="*40)
    
    img_train_dir = os.path.join(OUTPUT_DIR, "images", "train")
    lbl_train_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)

    h5_files = glob.glob(os.path.join(TEST_DIR, '**', '*.h5'), recursive=True)
    if not h5_files:
        print("❌ Erreur : Aucun fichier .h5 trouvé dans", TEST_DIR)
        return False

    print(f"🔄 Lecture de {len(h5_files)} fichiers H5 en cours...")
    img_count = 0

    for h5_path in h5_files:
        bbox_file = find_bbox_file(h5_path)
        if not bbox_file: 
            continue
        
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'data' not in f: continue
                ds = f['data']
                T_len = ds.shape[0]
                if T_len == 0: continue
                
                indices = np.linspace(0, T_len - 1, min(FRAMES_PER_FILE, T_len), dtype=int)
                dt = int(ds.attrs.get('delta_t', 50000))
                orig_w = int(ds.attrs.get('event_input_width', GEN4_W))
                orig_h = int(ds.attrs.get('event_input_height', GEN4_H))
                bboxes_all = np.load(bbox_file)
                
                for idx in indices:
                    frame = ds[idx]
                    
                    # Reconstruction image
                    if frame.shape[0] < frame.shape[-1]:
                        C = frame.shape[0]; H_img, W_img = frame.shape[1], frame.shape[2]
                        neg = np.sum(frame[0::2], axis=0) if C > 1 else np.sum(frame, axis=0)
                        pos = np.sum(frame[1::2], axis=0) if C > 1 else np.zeros_like(neg)
                    else:
                        H_img, W_img = frame.shape[0], frame.shape[1]; C = frame.shape[-1]
                        neg = np.sum(frame[..., 0::2], axis=-1) if C > 1 else np.sum(frame, axis=-1)
                        pos = np.sum(frame[..., 1::2], axis=-1) if C > 1 else np.zeros_like(neg)
                    
                    img = np.zeros((H_img, W_img, 3), dtype=np.float32)
                    img[:, :, 2] = pos # Rouge
                    img[:, :, 0] = neg # Bleu
                    
                    mx = max(pos.max(), neg.max())
                    if mx > 0: img = (img / mx * 255).astype(np.uint8)
                    else: img = img.astype(np.uint8)

                    # Boîtes YOLO
                    t0 = idx * dt
                    mask = (bboxes_all['t'] >= t0) & (bboxes_all['t'] <= t0 + dt)
                    b_win = bboxes_all[mask]
                    
                    txt_content = ""
                    if len(b_win) > 0:
                        best_t = np.max(b_win['t'])
                        for b in b_win[b_win['t'] == best_t]:
                            x_center = (b['x'] + b['w'] / 2.0) / float(orig_w)
                            y_center = (b['y'] + b['h'] / 2.0) / float(orig_h)
                            w_norm = b['w'] / float(orig_w)
                            h_norm = b['h'] / float(orig_h)
                            class_id = int(b['class_id'])
                            # On s'assure que les valeurs ne débordent pas
                            if 0 < x_center < 1 and 0 < y_center < 1:
                                txt_content += f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    
                    # On sauvegarde TOUJOURS l'image (même si txt_content est vide)
                    base_name_img = f"event_frame_{img_count}"
                    cv2.imwrite(os.path.join(img_train_dir, f"{base_name_img}.jpg"), img)
                    with open(os.path.join(lbl_train_dir, f"{base_name_img}.txt"), "w") as f_txt:
                        f_txt.write(txt_content)
                    
                    img_count += 1
                    
        except Exception as e:
            # SI ÇA PLANTE, ON VEUT LE SAVOIR !
            print(f"⚠️ Erreur lors du traitement de {os.path.basename(h5_path)} : {e}")

    print(f"✅ Succès ! {img_count} images extraites pour YOLO.")
    return img_count > 0

def phase2_create_yaml():
    print("\n" + "="*40)
    print("📝 PHASE 2 : Création du fichier YAML")
    print("="*40)
    abs_path = os.path.abspath(OUTPUT_DIR).replace('\\', '/')
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/train

nc: 7
names: ['classe_0', 'classe_1', 'classe_2', 'classe_3', 'classe_4', 'classe_5', 'classe_6']
"""
    with open(YAML_FILE, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"✅ Fichier '{YAML_FILE}' généré.")

def phase3_train_yolo():
    print("\n" + "="*40)
    print("🧠 PHASE 3 : Entraînement YOLOv8")
    print("="*40)
    model = YOLO("yolov8n.pt") 
    results = model.train(
        data=YAML_FILE, 
        epochs=EPOCHS, 
        imgsz=640, 
        batch=BATCH_SIZE, 
        device=0, 
        plots=True,
        project="P6_YOLO",
        name="run_events"
    )
    print("\n🎉 ENTRAÎNEMENT TERMINÉ ! Regarde le dossier 'P6_YOLO/run_events'.")

if __name__ == '__main__':
    # Si la phase 1 réussit (plus de 0 image), on lance la suite !
    if phase1_prepare_data():
        phase2_create_yaml()
        phase3_train_yolo()
    else:
        print("❌ L'entraînement a été annulé car aucune image n'a été trouvée.")