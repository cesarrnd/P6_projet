import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import numpy as np
import h5py
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from mean_average_precision import MetricBuilder

NUM_CLASSES = 8
TEST_DIR = r"D:\Cours 2025-2026\P6 Projet event camera\test"
MODEL_PATH = "model_event_cam.pth"
GEN4_W, GEN4_H = 1280, 720

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

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Calcul de la précision (mAP) avec MetricBuilder... Patientez.")

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Initialisation du nouvel outil de calcul (sans COCO !)
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=NUM_CLASSES)

    h5_files = glob.glob(os.path.join(TEST_DIR, '**', '*.h5'), recursive=True)
    np.random.shuffle(h5_files)
    test_files = h5_files[:20] 

    for idx_file, h5_path in enumerate(test_files):
        bbox_file = find_bbox_file(h5_path)
        if not bbox_file: continue
        
        with h5py.File(h5_path, 'r') as f:
            ds = f['data']
            if ds.shape[0] == 0: continue
            frame_idx = ds.shape[0] // 2
            frame = ds[frame_idx]
            delta_t = int(ds.attrs.get('delta_t', 50000))
            orig_w = int(ds.attrs.get('event_input_width', GEN4_W))
            orig_h = int(ds.attrs.get('event_input_height', GEN4_H))

        # 1. Formatage Image
        if frame.shape[0] < frame.shape[-1]:
            C = frame.shape[0]; H_img, W_img = frame.shape[1], frame.shape[2]
            neg = np.sum(frame[0::2], axis=0) if C > 1 else np.sum(frame, axis=0)
            pos = np.sum(frame[1::2], axis=0) if C > 1 else np.zeros_like(neg)
        else:
            H_img, W_img = frame.shape[0], frame.shape[1]; C = frame.shape[-1]
            neg = np.sum(frame[..., 0::2], axis=-1) if C > 1 else np.sum(frame, axis=-1)
            pos = np.sum(frame[..., 1::2], axis=-1) if C > 1 else np.zeros_like(neg)

        img = np.zeros((3, H_img, W_img), dtype=np.float32)
        img[0, :, :] = pos; img[2, :, :] = neg
        max_val = max(pos.max(), neg.max())
        if max_val > 0: img = np.clip((img / max_val), 0, 1)

        # 2. Formatage Vraies Boîtes (Ground Truth)
        t0 = frame_idx * delta_t
        bboxes_all = np.load(bbox_file)
        mask = (bboxes_all['t'] >= t0) & (bboxes_all['t'] <= t0 + delta_t)
        b_in_window = bboxes_all[mask]
        
        gt_boxes = []
        if len(b_in_window) > 0:
            scale_x = W_img / float(orig_w); scale_y = H_img / float(orig_h)
            for b in b_in_window[b_in_window['t'] == np.max(b_in_window['t'])]:
                xmin = b['x'] * scale_x; ymin = b['y'] * scale_y
                xmax = xmin + (b['w'] * scale_x); ymax = ymin + (b['h'] * scale_y)
                if xmax > xmin and ymax > ymin:
                    # Format demandé : [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                    gt_boxes.append([xmin, ymin, xmax, ymax, b['class_id'], 0, 0])
        
        if len(gt_boxes) == 0: continue
        gt_numpy = np.array(gt_boxes)

        # 3. Prédiction
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)[0]
            
        # 4. Formatage des prédictions
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy() - 1 # On retire le +1 du background
        
        preds = []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            xmin, ymin, xmax, ymax = box
            # Format demandé : [xmin, ymin, xmax, ymax, class_id, confidence]
            preds.append([xmin, ymin, xmax, ymax, label, score])
            
        preds_numpy = np.array(preds) if len(preds) > 0 else np.empty((0, 6))

        # 5. Ajout à la métrique
        metric_fn.add(preds_numpy, gt_numpy)
        torch.cuda.empty_cache()

    # --- RESULTATS ---
    print("\n--- RÉSULTATS OFFICIELS DE VOTRE MODÈLE ---")
    # On demande de calculer avec un IoU de 50% (0.5)
    print(f"mAP@50 (Prédictions correctes à 50% d'IoU) : {metric_fn.value(iou_thresholds=0.5)['mAP']:.4f}")

if __name__ == '__main__':
    main()