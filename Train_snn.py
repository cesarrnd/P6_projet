"""
=============================================================================
SNN EVENT CAMERA OBJECT DETECTOR — VERSION CORRIGÉE
Inspiré de : "Brain-Inspired Spiking Neural Networks for Energy-Efficient
Object Detection" (Li et al., CVPR 2025) — Architecture MSD / ONNB

Caméra : Prophesee Gen4 (1280×720)
Framework : PyTorch + SpikingJelly

CORRECTIONS APPLIQUÉES (vs version originale) :
  FIX-1  decode_predictions : offset de cellule (gi, gj) ajouté → cx/cy corrects
  FIX-2  SpikeYOLOLoss      : anchors[best_a] au lieu de anchors[0] pour w/h
  FIX-3  SpikeYOLOLoss      : _best_anchor accepte float Python (torch.tensor wrap)
  FIX-4  compute_map50      : NMS appliquée avant le calcul → faux positifs éliminés
  FIX-5  Dataset            : fenêtre temporelle GT élargie ±10% → moins de GT perdus
  FIX-6  Optimiseur         : SGD → AdamW + OneCycleLR (warmup 10%) → stabilité BPTT
  FIX-7  Hyperparamètres    : MAX_FILES/WIN augmentés, résolution 180×320, T_STEPS=8
  FIX-8  Anchors            : utilitaire k-means intégré pour calibrer sur tes données
=============================================================================
"""

import os
import math
import time
import random
import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import GradScaler, autocast          # PyTorch >= 2.0
except ImportError:
    from torch.cuda.amp import GradScaler, autocast     # PyTorch < 2.0

from spikingjelly.activation_based import neuron, functional, surrogate, layer

# ===========================================================================
#  MODE — mettre SMOKE_TEST = True pour valider le pipeline en ~5 min
#          mettre SMOKE_TEST = False pour le vrai entrainement
# ===========================================================================

SMOKE_TEST = True   # <- changer ici uniquement

# ===========================================================================
#  CONSTANTES — MODIFIEZ CES VALEURS SELON VOTRE CONFIGURATION
# ===========================================================================

DATA_ROOT      = r"D:\Cours 2025-2026\P6 Projet event camera"
TRAIN_DIR      = os.path.join(DATA_ROOT, "train")
VAL_DIR        = os.path.join(DATA_ROOT, "val")

RAW_H, RAW_W   = 720, 1280
NUM_CLASSES    = 7

ANCHORS = [
    [(0.028, 0.07), (0.057, 0.11), (0.083, 0.17)],
    [(0.111, 0.24), (0.167, 0.31), (0.222, 0.43)],
    [(0.306, 0.56), (0.389, 0.72), (0.500, 0.88)],
]

WEIGHT_DECAY         = 1e-4
GRAD_CLIP            = 1.0
USE_AMP              = True
TAU                  = 0.25
V_THRESHOLD          = 0.5
V_RESET              = 0.0
ALPHA_TdBN           = 1.0
NUM_WORKERS          = 0
PIN_MEMORY           = True
SEED                 = 42
GT_TIME_MARGIN_RATIO = 0.10

# ---------------------------------------------------------------------------
# Parametres qui changent entre smoke test et production
# ---------------------------------------------------------------------------
if SMOKE_TEST:
    # SMOKE TEST : ~5 min sur GTX 1050
    # Objectif : verifier que les 3 losses descendent et que mAP > 0
    CHECKPOINT_DIR   = os.path.join(DATA_ROOT, "checkpoints_smoke")
    INPUT_H          = 90        # resolution basse -> forward 4x plus rapide
    INPUT_W          = 160
    T_STEPS          = 4         # moins de bins -> BPTT plus court
    DELTA_T_US       = 50_000
    BATCH_SIZE       = 2         # on peut monter car resolution reduite
    EPOCHS           = 3         # 3 epoques suffisent pour tout detecter
    LR_MAX           = 3e-4
    MAX_FILES_TRAIN  = 5         # ~250 samples, chargement quasi-instantane
    MAX_FILES_VAL    = 2
    MAX_WIN_PER_FILE = 50
    MAP_EVAL_EVERY   = 1         # mAP a chaque epoque pour voir vite
else:
    # PRODUCTION : vrai entrainement
    CHECKPOINT_DIR   = os.path.join(DATA_ROOT, "checkpoints_snn_fixed")
    INPUT_H          = 180
    INPUT_W          = 320
    T_STEPS          = 8
    DELTA_T_US       = 50_000
    BATCH_SIZE       = 1
    EPOCHS           = 80
    LR_MAX           = 3e-4
    MAX_FILES_TRAIN  = 28      # tous les fichiers disponibles
    MAX_FILES_VAL    = 7
    MAX_WIN_PER_FILE = 10
    MAP_EVAL_EVERY   = 5


# ===========================================================================
# 0. UTILITAIRE : calibration des anchors par k-means (FIX-8)
# ===========================================================================

def compute_anchors_kmeans(data_dir: str, n_anchors: int = 9,
                           max_files: int = 50) -> list:
    """
    Calcule les anchors optimales pour ton dataset par k-means sur les
    dimensions (w, h) normalisées de toutes les boîtes GT.

    Retourne une liste de 3 listes de (w, h), triées par aire croissante,
    prête à être copiée dans ANCHORS.

    Usage :
        anchors = compute_anchors_kmeans(TRAIN_DIR)
        print(anchors)   # copier-coller dans ANCHORS ci-dessus
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("  ⚠ sklearn non installé : pip install scikit-learn")
        return ANCHORS

    all_wh = []
    fnames = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
    random.shuffle(fnames)
    fnames = fnames[:max_files]

    for fname in fnames:
        npy_path = EventSpikeDataset._find_npy(data_dir, fname)
        if npy_path is None:
            continue
        raw = np.load(npy_path)
        if raw.dtype.names:
            try:
                with h5py.File(os.path.join(data_dir, fname), "r") as f:
                    orig_w = int(f["data"].attrs.get("event_input_width",  RAW_W))
                    orig_h = int(f["data"].attrs.get("event_input_height", RAW_H))
            except Exception:
                orig_w, orig_h = RAW_W, RAW_H
            for r in raw:
                bw = float(r["w"]) / orig_w
                bh = float(r["h"]) / orig_h
                if bw > 0 and bh > 0:
                    all_wh.append([bw, bh])

    if len(all_wh) < n_anchors:
        print(f"  ⚠ Seulement {len(all_wh)} boîtes trouvées, anchors inchangées.")
        return ANCHORS

    boxes  = np.array(all_wh)
    km     = KMeans(n_clusters=n_anchors, n_init=20, random_state=42).fit(boxes)
    cents  = sorted(km.cluster_centers_.tolist(), key=lambda x: x[0] * x[1])
    result = [
        [tuple(cents[i]) for i in range(0, 3)],
        [tuple(cents[i]) for i in range(3, 6)],
        [tuple(cents[i]) for i in range(6, 9)],
    ]
    print("  Anchors k-means calculées :")
    for i, scale in enumerate(result):
        print(f"    Échelle {i} : {scale}")
    return result


# ===========================================================================
# 1. DATASET
# ===========================================================================

class EventSpikeDataset(Dataset):
    """
    Lit les fichiers .h5 Prophesee et les .npy bbox associés.

    FIX-5 : la fenêtre temporelle de filtrage des GT est élargie de ±10%
            pour éviter de perdre des labels dont le timestamp borde la fenêtre.
    """

    def __init__(self, data_dir: str, t_steps: int = T_STEPS,
                 delta_t_us: int = DELTA_T_US,
                 h: int = INPUT_H, w: int = INPUT_W,
                 max_files: int = None):
        self.t_steps  = t_steps
        self.delta_t  = delta_t_us
        self.h        = h
        self.w        = w
        self.samples  = []

        all_fnames = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
        if max_files and len(all_fnames) > max_files:
            random.shuffle(all_fnames)
            all_fnames = all_fnames[:max_files]

        for fname in all_fnames:
            h5_path  = os.path.join(data_dir, fname)
            npy_path = self._find_npy(data_dir, fname)
            if npy_path is None:
                continue
            self._index_file(h5_path, npy_path)

        print(f"  Dataset '{os.path.basename(data_dir)}' : "
              f"{len(self.samples)} fenêtres "
              f"({len(all_fnames)} fichiers)")

    @staticmethod
    def _find_npy(data_dir: str, h5_fname: str):
        import glob as _glob
        direct = os.path.join(data_dir, h5_fname.replace(".h5", "_bbox.npy"))
        if os.path.exists(direct):
            return direct
        base   = h5_fname.replace(".h5", "")
        parts  = base.split("_")
        stamps = [p for p in parts if p.isdigit() and len(p) >= 8]
        if len(stamps) >= 2:
            cands = _glob.glob(os.path.join(
                data_dir, f"*{stamps[0]}*{stamps[1]}*bbox*.npy"))
            if cands:
                return cands[0]
        return None

    def _index_file(self, h5_path: str, npy_path: str):
        try:
            with h5py.File(h5_path, "r") as f:
                ds        = f["data"]
                N_frames  = ds.shape[0]
                dt_frame  = int(ds.attrs.get("delta_t",
                                             self.delta_t // self.t_steps))
                orig_w    = int(ds.attrs.get("event_input_width",  RAW_W))
                orig_h    = int(ds.attrs.get("event_input_height", RAW_H))
        except (KeyError, OSError) as e:
            print(f"  ⚠ Ignoré ({os.path.basename(h5_path)}) : {e}")
            return

        n_windows  = N_frames // self.t_steps
        all_starts = list(range(n_windows))
        if MAX_WIN_PER_FILE and len(all_starts) > MAX_WIN_PER_FILE:
            all_starts = random.sample(all_starts, MAX_WIN_PER_FILE)
        for i in all_starts:
            self.samples.append((
                h5_path, npy_path,
                i * self.t_steps,
                orig_w, orig_h, dt_frame,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        h5_path, npy_path, frame_start, orig_w, orig_h, dt_frame = \
            self.samples[idx]

        # ── 1. Lecture frames H5 ─────────────────────────────────────────
        with h5py.File(h5_path, "r") as f:
            frames = f["data"][frame_start: frame_start + self.t_steps]

        T, C, H0, W0 = frames.shape
        spikes_raw = np.zeros((T, 2, H0, W0), dtype=np.float32)

        if C == 1:
            spikes_raw[:, 0] = frames[:, 0].astype(np.float32)
        elif C == 2:
            spikes_raw[:, 1] = frames[:, 0].astype(np.float32)
            spikes_raw[:, 0] = frames[:, 1].astype(np.float32)
        else:
            neg_idx = list(range(0, C, 2))
            pos_idx = list(range(1, C, 2))
            for t in range(T):
                spikes_raw[t, 1] = frames[t][neg_idx].astype(np.float32).sum(0)
                spikes_raw[t, 0] = frames[t][pos_idx].astype(np.float32).sum(0)

        mx = spikes_raw.max()
        if mx > 0:
            spikes_raw /= mx

        if H0 != self.h or W0 != self.w:
            flat = spikes_raw.reshape(T * 2, H0, W0)
            resized = np.stack([
                cv2.resize(flat[i], (self.w, self.h),
                           interpolation=cv2.INTER_AREA)
                for i in range(T * 2)
            ], axis=0)
            spikes_raw = resized.reshape(T, 2, self.h, self.w)

        spikes = torch.from_numpy(spikes_raw)

        # ── 2. Filtrage temporel des GT (FIX-5 : marge ±10%) ────────────
        t_start_us = frame_start * dt_frame
        t_end_us   = t_start_us + self.t_steps * dt_frame
        # FIX-5 : marge pour ne pas perdre les labels aux bords de fenêtre
        margin = int(GT_TIME_MARGIN_RATIO * self.t_steps * dt_frame)

        raw_labels = np.load(npy_path)
        if raw_labels.dtype.names and "t" in raw_labels.dtype.names:
            lt     = raw_labels["t"].astype(np.int64)
            mask_l = (lt >= t_start_us - margin) & (lt < t_end_us + margin)
            raw_labels = raw_labels[mask_l]

        targets = self._parse_labels(raw_labels, orig_w, orig_h)
        return spikes, targets

    def _parse_labels(self, raw, orig_w: int, orig_h: int) -> torch.Tensor:
        if len(raw) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        rows = []
        for r in raw:
            if raw.dtype.names:
                cls = int(r["class_id"])
                x1  = float(r["x"]) / orig_w
                y1  = float(r["y"]) / orig_h
                bw  = float(r["w"]) / orig_w
                bh  = float(r["h"]) / orig_h
            else:
                cls = int(r[5])
                x1  = float(r[1]) / orig_w
                y1  = float(r[2]) / orig_h
                bw  = float(r[3]) / orig_w
                bh  = float(r[4]) / orig_h
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
            if bw > 0 and bh > 0:
                rows.append([cls, cx, cy, bw, bh])
        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)


def collate_fn(batch):
    spikes  = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return spikes, targets


class CachedDataset(Dataset):
    def __init__(self, dataset: EventSpikeDataset):
        print(f"  → Pré-chargement RAM de {len(dataset)} échantillons...",
              end=" ", flush=True)
        self.data = [dataset[i] for i in range(len(dataset))]
        mem_mb = sum(s.nbytes for s, _ in self.data) / 1e6
        print(f"OK ({mem_mb:.0f} Mo)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ===========================================================================
# 2. ARCHITECTURE SNN
# ===========================================================================

class tdBN(nn.Module):
    def __init__(self, num_features: int, alpha: float = ALPHA_TdBN,
                 v_th: float = V_THRESHOLD, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.v_th  = v_th
        self.bn    = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)
        with torch.no_grad():
            self.bn.weight.fill_(alpha * v_th)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.bn(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class SCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias)
        self.tbn  = tdBN(out_channels)
        self.lif  = neuron.LIFNode(
            tau=1.0 / TAU, v_threshold=V_THRESHOLD, v_reset=V_RESET,
            surrogate_function=surrogate.ATan(), detach_reset=True,
            step_mode='m',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        x_flat = self.conv(x_flat)
        _, C2, H2, W2 = x_flat.shape
        x = x_flat.reshape(B, T, C2, H2, W2)
        x = self.tbn(x)
        x = x.permute(1, 0, 2, 3, 4)
        x = self.lif(x)
        x = x.permute(1, 0, 2, 3, 4)
        return x


class SpikeNeuronUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scn1  = SCN(channels, channels)
        self.scn2  = SCN(channels, channels)
        self.pool  = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.scn_s = SCN(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.scn2(self.scn1(x))
        B, T, C, H, W = x.shape
        xp    = x.reshape(B * T, C, H, W)
        xp    = self.pool(xp).reshape(B, T, C, H, W)
        short = self.scn_s(xp)
        return torch.cat([main, short], dim=2)


class ONNB(nn.Module):
    def __init__(self, in_c: int, out_c: int,
                 n_sn: int = 2, stride: int = 1):
        super().__init__()
        mid_c = out_c // 2
        self.stem    = SCN(in_c, out_c, stride=stride, padding=1)
        self.n_sn    = n_sn
        self.sn_list = nn.ModuleList()
        c_in = mid_c
        for _ in range(n_sn):
            self.sn_list.append(SpikeNeuronUnit(c_in))
            c_in *= 2
        total_c        = mid_c + c_in
        self.fuse_conv = SCN(total_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        B, T, C, H, W = x.shape
        mid      = C // 2
        x_direct = x[:, :, :mid, :, :]
        x_sn     = x[:, :, mid:, :, :]
        for sn in self.sn_list:
            x_sn = sn(x_sn)
        return self.fuse_conv(torch.cat([x_direct, x_sn], dim=2))


class SPPFSpiking(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid_c     = in_c // 2
        self.cv1  = SCN(in_c, mid_c, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.cv2  = SCN(mid_c * 4, out_c, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        B, T, C, H, W = x.shape
        xf = x.reshape(B * T, C, H, W)
        y1 = self.pool(xf)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        def up(t): return F.interpolate(t, size=(H, W), mode="nearest")
        x_cat = torch.cat([xf, up(y1), up(y2), up(y3)], dim=1)
        return self.cv2(x_cat.reshape(B, T, C * 4, H, W))


class SNNBackbone(nn.Module):
    def __init__(self, in_c: int = 2):
        super().__init__()
        self.stage0 = SCN(in_c, 8)
        self.stage1 = nn.Sequential(SCN(8, 16, stride=2, padding=1),
                                     ONNB(16, 16, n_sn=1))
        self.stage2 = nn.Sequential(SCN(16, 32, stride=2, padding=1),
                                     ONNB(32, 32, n_sn=1))
        self.stage3 = nn.Sequential(SCN(32, 64, stride=2, padding=1),
                                     ONNB(64, 64, n_sn=1))
        self.stage4 = nn.Sequential(SCN(64, 128, stride=2, padding=1),
                                     ONNB(128, 128, n_sn=1))
        self.sppf   = SPPFSpiking(128, 128)

    def forward(self, x: torch.Tensor):
        x  = self.stage0(x)
        x  = self.stage1(x)
        x  = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.sppf(c4)
        return c3, c4, c5


class SPNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat5  = SCN(128, 64, kernel_size=1, padding=0)
        self.fuse4 = ONNB(192, 64, n_sn=1)
        self.lat4  = SCN(64, 32, kernel_size=1, padding=0)
        self.fuse3 = ONNB(96, 32, n_sn=1)
        self.down3 = SCN(32, 64, stride=2, padding=1)
        self.pan4  = ONNB(128, 64, n_sn=1)
        self.down4 = SCN(64, 128, stride=2, padding=1)
        self.pan5  = ONNB(256, 128, n_sn=1)

    def _up(self, x: torch.Tensor, size) -> torch.Tensor:
        B, T, C, H, W = x.shape
        xf = x.reshape(B * T, C, H, W)
        xf = F.interpolate(xf, size=size, mode="nearest")
        return xf.reshape(B, T, C, *xf.shape[2:])

    def forward(self, c3, c4, c5):
        p5     = self.lat5(c5)
        _, _, _, h4, w4 = c4.shape
        p4     = self.fuse4(torch.cat([c4, self._up(p5, (h4, w4))], dim=2))
        _, _, _, h3, w3 = c3.shape
        p3     = self.fuse3(torch.cat([c3, self._up(self.lat4(p4),
                                                     (h3, w3))], dim=2))
        p3d    = self.down3(p3)
        _, _, _, h4_, w4_ = p4.shape
        n4     = self.pan4(torch.cat([p4, self._up(p3d, (h4_, w4_))], dim=2))
        n4d    = self.down4(n4)
        _, _, _, h5_, w5_ = c5.shape
        n5     = self.pan5(torch.cat([c5, self._up(n4d, (h5_, w5_))], dim=2))
        return p3, n4, n5


class SpikingDetectHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.na      = num_anchors
        self.nc      = num_classes
        self.cls_scn = SCN(in_channels, in_channels)
        self.cls_out = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.box_scn = SCN(in_channels, in_channels)
        self.box_out = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_scn = SCN(in_channels, in_channels)
        self.obj_out = nn.Conv2d(in_channels, num_anchors, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        cls_feat = self.cls_scn(x).mean(dim=1)
        box_feat = self.box_scn(x).mean(dim=1)
        obj_feat = self.obj_scn(x).mean(dim=1)
        cls_out  = self.cls_out(cls_feat).permute(0,2,3,1).reshape(
                       B, H, W, self.na, self.nc)
        box_out  = self.box_out(box_feat).permute(0,2,3,1).reshape(
                       B, H, W, self.na, 4)
        obj_out  = self.obj_out(obj_feat).permute(0,2,3,1).reshape(
                       B, H, W, self.na, 1)
        return torch.cat([box_out, obj_out, cls_out], dim=-1)


class MSDSpikeDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 anchors: list = ANCHORS):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_def = anchors
        na = len(anchors[0])
        self.backbone = SNNBackbone(in_c=2)
        self.neck     = SPNeck()
        self.head_p3  = SpikingDetectHead(32,  na, num_classes)
        self.head_n4  = SpikingDetectHead(64,  na, num_classes)
        self.head_n5  = SpikingDetectHead(128, na, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                         nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        c3, c4, c5 = self.backbone(x)
        p3, n4, n5 = self.neck(c3, c4, c5)
        return [self.head_p3(p3), self.head_n4(n4), self.head_n5(n5)]


# ===========================================================================
# 3. LOSS — corrections FIX-2 et FIX-3
# ===========================================================================

class SpikeYOLOLoss(nn.Module):
    """
    FIX-2 : pred_decoded utilise anchors[best_a] (la bonne ancre)
             au lieu de anchors[0] en dur.
    FIX-3 : _best_anchor wrappe bw/bh en tenseur pour éviter les erreurs
             de type avec torch.min(float, Tensor).
    """

    def __init__(self, anchors: list = ANCHORS,
                 num_classes: int = NUM_CLASSES,
                 input_h: int = INPUT_H, input_w: int = INPUT_W,
                 strides: list = None):
        super().__init__()
        self.nc      = num_classes
        self.strides = strides or [8, 16, 32]
        self.anchors = anchors
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
        self.lambda_box = 0.05
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    @staticmethod
    def ciou_loss(pred_boxes: torch.Tensor,
                  tgt_boxes: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        t_x1 = tgt_boxes[:, 0] - tgt_boxes[:, 2] / 2
        t_y1 = tgt_boxes[:, 1] - tgt_boxes[:, 3] / 2
        t_x2 = tgt_boxes[:, 0] + tgt_boxes[:, 2] / 2
        t_y2 = tgt_boxes[:, 1] + tgt_boxes[:, 3] / 2
        ix1  = torch.max(p_x1, t_x1)
        iy1  = torch.max(p_y1, t_y1)
        ix2  = torch.min(p_x2, t_x2)
        iy2  = torch.min(p_y2, t_y2)
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
        t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
        iou   = inter / (p_area + t_area - inter + eps)
        c_px  = (p_x1 + p_x2) / 2
        c_py  = (p_y1 + p_y2) / 2
        c_tx  = (t_x1 + t_x2) / 2
        c_ty  = (t_y1 + t_y2) / 2
        rho2  = (c_px - c_tx) ** 2 + (c_py - c_ty) ** 2
        enc_x1 = torch.min(p_x1, t_x1)
        enc_y1 = torch.min(p_y1, t_y1)
        enc_x2 = torch.max(p_x2, t_x2)
        enc_y2 = torch.max(p_y2, t_y2)
        c2    = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps
        v     = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(tgt_boxes[:, 2] / (tgt_boxes[:, 3] + eps)) -
            torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps)), 2)
        alpha = v / (1 - iou + v + eps)
        return (1 - iou + rho2 / c2 + alpha * v).mean()

    def forward(self, predictions: list,
                targets: list) -> tuple:
        device    = predictions[0].device
        loss_tot  = torch.zeros(1, device=device)
        loss_dict = {"box": 0., "obj": 0., "cls": 0.}
        B = predictions[0].shape[0]

        for scale_idx, pred in enumerate(predictions):
            B, H, W, na, _ = pred.shape
            anchors = torch.tensor(self.anchors[scale_idx],
                                   dtype=torch.float32, device=device)

            obj_tgt  = torch.zeros(B, H, W, na, device=device)
            cls_tgt  = torch.zeros(B, H, W, na, self.nc, device=device)
            box_tgt  = torch.zeros(B, H, W, na, 4, device=device)
            box_mask = torch.zeros(B, H, W, na, dtype=torch.bool, device=device)
            # FIX-2 : stocker l'indice d'ancre par cellule pour le décodage
            anch_idx = torch.zeros(B, H, W, na, dtype=torch.long, device=device)

            for b_idx, tgt in enumerate(targets):
                if tgt.shape[0] == 0:
                    continue
                tgt = tgt.to(device)
                for obj in tgt:
                    cls_id, cx, cy, bw, bh = obj
                    # FIX-3 : wrapper bw/bh en tenseur avant _best_anchor
                    best_a = self._best_anchor(
                        torch.tensor(bw.item()),
                        torch.tensor(bh.item()),
                        anchors
                    )
                    gi = max(0, min(int(cx * W), W - 1))
                    gj = max(0, min(int(cy * H), H - 1))
                    obj_tgt[b_idx, gj, gi, best_a]              = 1.0
                    cls_tgt[b_idx, gj, gi, best_a, int(cls_id)] = 1.0
                    box_tgt[b_idx, gj, gi, best_a]              = obj[1:]
                    box_mask[b_idx, gj, gi, best_a]             = True
                    anch_idx[b_idx, gj, gi, best_a]             = best_a

            l_obj = self.bce(pred[..., 4], obj_tgt)

            if box_mask.any():
                pred_cls = pred[..., 5:][box_mask]
                tgt_cls  = cls_tgt[box_mask]
                l_cls    = self.bce(pred_cls, tgt_cls)

                pred_box = pred[..., :4][box_mask]    # [N_pos, 4]
                ai       = anch_idx[box_mask]         # [N_pos]

                # FIX-2 : utiliser la bonne ancre pour chaque détection
                aw = anchors[ai, 0]   # [N_pos]
                ah = anchors[ai, 1]   # [N_pos]

                pred_decoded = torch.zeros_like(pred_box)
                pred_decoded[:, 0] = torch.sigmoid(pred_box[:, 0])
                pred_decoded[:, 1] = torch.sigmoid(pred_box[:, 1])
                pred_decoded[:, 2] = torch.exp(pred_box[:, 2].clamp(-4, 4)) * aw
                pred_decoded[:, 3] = torch.exp(pred_box[:, 3].clamp(-4, 4)) * ah

                l_box = self.ciou_loss(pred_decoded, box_tgt[box_mask])
            else:
                l_cls = torch.zeros(1, device=device)
                l_box = torch.zeros(1, device=device)

            scale_loss = (self.lambda_obj * l_obj +
                          self.lambda_cls * l_cls +
                          self.lambda_box * l_box)
            loss_tot  += scale_loss
            loss_dict["obj"] += l_obj.item()
            loss_dict["cls"] += l_cls.item()
            loss_dict["box"] += l_box.item()

        return loss_tot, loss_dict

    @staticmethod
    def _best_anchor(bw: torch.Tensor, bh: torch.Tensor,
                     anchors: torch.Tensor) -> int:
        # FIX-3 : s'assurer que bw/bh sont des tenseurs scalaires
        bw = bw.float() if isinstance(bw, torch.Tensor) else torch.tensor(float(bw))
        bh = bh.float() if isinstance(bh, torch.Tensor) else torch.tensor(float(bh))
        inter = (torch.min(bw, anchors[:, 0]) *
                 torch.min(bh, anchors[:, 1]))
        union = bw * bh + anchors[:, 0] * anchors[:, 1] - inter + 1e-7
        return int((inter / union).argmax())


# ===========================================================================
# 4. DÉCODAGE CORRIGÉ (FIX-1) + NMS (FIX-4)
# ===========================================================================

@torch.no_grad()
def decode_predictions(predictions: list, anchors: list,
                       conf_thresh: float = 0.05):
    """
    FIX-1 : cx/cy incluent l'offset (col, row) de la cellule dans la grille
             et sont normalisés par W/H.
             Dans l'original, l'offset était absent → toutes les boîtes
             pointaient vers (0,0) → IoU GT ≈ 0 → mAP = 0.
    """
    device  = predictions[0].device
    B       = predictions[0].shape[0]
    results = [[] for _ in range(B)]

    for scale_idx, pred in enumerate(predictions):
        _, H, W, na, _ = pred.shape
        anch = torch.tensor(anchors[scale_idx],
                            dtype=torch.float32, device=device)

        # FIX-1 : grilles d'offset pour chaque cellule
        grid_y = torch.arange(H, dtype=torch.float32, device=device)
        grid_x = torch.arange(W, dtype=torch.float32, device=device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        gx = gx.view(1, H, W, 1)   # [1, H, W, 1] → broadcast sur B et na
        gy = gy.view(1, H, W, 1)

        # FIX-1 : (sigmoid + offset) / taille_grille
        cx = (torch.sigmoid(pred[..., 0]) + gx) / W
        cy = (torch.sigmoid(pred[..., 1]) + gy) / H

        # FIX-2 bis : ancres correctement broadcastées par indice
        aw = anch[:, 0].view(1, 1, 1, na)
        ah = anch[:, 1].view(1, 1, 1, na)
        bw = torch.exp(pred[..., 2].clamp(-4, 4)) * aw
        bh = torch.exp(pred[..., 3].clamp(-4, 4)) * ah

        obj       = torch.sigmoid(pred[..., 4])
        cls_prob  = torch.sigmoid(pred[..., 5:])
        cls_score, cls_id = cls_prob.max(dim=-1)
        conf      = obj * cls_score

        mask = conf > conf_thresh
        for b in range(B):
            m = mask[b]
            if not m.any():
                continue
            boxes = torch.stack([
                cx[b][m], cy[b][m], bw[b][m], bh[b][m],
                conf[b][m], cls_id[b][m].float()
            ], dim=-1)
            results[b].append(boxes)

    final = []
    for b in range(B):
        if results[b]:
            final.append(torch.cat(results[b], dim=0))
        else:
            final.append(torch.zeros((0, 6), device=device))
    return final


def _batch_iou_cx(box1: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    b1x1 = box1[:, 0] - box1[:, 2] / 2
    b1y1 = box1[:, 1] - box1[:, 3] / 2
    b1x2 = box1[:, 0] + box1[:, 2] / 2
    b1y2 = box1[:, 1] + box1[:, 3] / 2
    bx1  = boxes[:, 0] - boxes[:, 2] / 2
    by1  = boxes[:, 1] - boxes[:, 3] / 2
    bx2  = boxes[:, 0] + boxes[:, 2] / 2
    by2  = boxes[:, 1] + boxes[:, 3] / 2
    ix   = (torch.min(b1x2, bx2) - torch.max(b1x1, bx1)).clamp(0)
    iy   = (torch.min(b1y2, by2) - torch.max(b1y1, by1)).clamp(0)
    inter = ix * iy
    union = (box1[:, 2] * box1[:, 3] + boxes[:, 2] * boxes[:, 3]
             - inter + 1e-7)
    return inter / union


def nms_per_class(dets: torch.Tensor,
                  iou_thresh: float = 0.45) -> torch.Tensor:
    """
    FIX-4 : NMS par classe sur les détections [N, 6] (cx,cy,w,h,conf,cls).
    Absent dans l'original → faux positifs massifs → mAP écrasé.
    """
    if dets.shape[0] == 0:
        return dets
    kept = []
    for c in dets[:, 5].unique():
        d = dets[dets[:, 5] == c]
        d = d[d[:, 4].argsort(descending=True)]
        while d.shape[0] > 0:
            kept.append(d[0])
            if d.shape[0] == 1:
                break
            ious = _batch_iou_cx(d[0:1, :4], d[1:, :4])
            d    = d[1:][ious < iou_thresh]
    return torch.stack(kept) if kept else torch.zeros((0, 6), device=dets.device)


# ===========================================================================
# 5. mAP@0.50 CORRIGÉ (FIX-4 : NMS intégrée)
# ===========================================================================

def compute_map50(all_preds: list, all_targets: list,
                  num_classes: int, iou_thresh: float = 0.5) -> float:
    """
    FIX-4 : NMS appliquée sur all_preds avant le calcul.
    Calcule le mAP@IoU=0.5 avec interpolation 11 points.
    """
    detections = {c: [] for c in range(num_classes)}
    n_gt       = {c: 0  for c in range(num_classes)}

    for preds_raw, targets in zip(all_preds, all_targets):
        # FIX-4 : NMS avant le calcul mAP
        preds = nms_per_class(preds_raw.to("cpu"))

        gt_by_cls = {}
        if targets.shape[0] > 0:
            for row in targets:
                c   = int(row[0].item())
                box = row[1:].cpu()
                gt_by_cls.setdefault(c, []).append(box)
                n_gt[c] += 1

        matched = {c: [False] * len(v) for c, v in gt_by_cls.items()}

        if preds.shape[0] == 0:
            continue

        order = preds[:, 4].argsort(descending=True)
        for idx in order:
            det  = preds[idx].cpu()
            c    = int(det[5].item())
            box  = det[:4]
            conf = det[4].item()
            gt_boxes = gt_by_cls.get(c, [])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                iou = _box_iou_scalar(box, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = (best_iou >= iou_thresh and best_j >= 0
                     and c in matched and not matched[c][best_j])
            if is_tp:
                matched[c][best_j] = True
                detections[c].append((conf, 1))
            else:
                detections[c].append((conf, 0))

    aps = []
    for c in range(num_classes):
        if n_gt[c] == 0:
            continue
        dets   = sorted(detections[c], key=lambda x: -x[0])
        if not dets:
            aps.append(0.0)
            continue
        tp_cum = np.cumsum([d[1] for d in dets]).astype(float)
        fp_cum = np.cumsum([1 - d[1] for d in dets]).astype(float)
        rec    = tp_cum / (n_gt[c] + 1e-7)
        prec   = tp_cum / (tp_cum + fp_cum + 1e-7)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p   = prec[rec >= t].max() if (rec >= t).any() else 0.0
            ap += p / 11
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def _box_iou_scalar(b1: torch.Tensor, b2: torch.Tensor) -> float:
    b1x1 = float(b1[0]) - float(b1[2]) / 2
    b1y1 = float(b1[1]) - float(b1[3]) / 2
    b1x2 = float(b1[0]) + float(b1[2]) / 2
    b1y2 = float(b1[1]) + float(b1[3]) / 2
    b2x1 = float(b2[0]) - float(b2[2]) / 2
    b2y1 = float(b2[1]) - float(b2[3]) / 2
    b2x2 = float(b2[0]) + float(b2[2]) / 2
    b2y2 = float(b2[1]) + float(b2[3]) / 2
    ix   = max(0., min(b1x2, b2x2) - max(b1x1, b2x1))
    iy   = max(0., min(b1y2, b2y2) - max(b1y1, b2y1))
    inter = ix * iy
    union = ((b1x2-b1x1)*(b1y2-b1y1) + (b2x2-b2x1)*(b2y2-b2y1) - inter)
    return inter / (union + 1e-7)


# ===========================================================================
# 6. ENTRAÎNEMENT
# ===========================================================================

def _warmup_lr(optimizer, epoch: int, warmup_epochs: int, max_lr: float):
    """Warmup linéaire : lr monte de max_lr/25 à max_lr sur warmup_epochs époques."""
    if epoch <= warmup_epochs:
        lr = (max_lr / 25.0) + (epoch - 1) / max(warmup_epochs - 1, 1) * (max_lr - max_lr / 25.0)
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, scaler, device, epoch: int) -> dict:
    model.train()
    total = {"loss": 0., "obj": 0., "cls": 0., "box": 0.}
    n     = len(loader)

    for batch_idx, (spikes, targets) in enumerate(loader):
        spikes = spikes.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=USE_AMP):
            predictions       = model(spikes)
            loss, loss_dict   = criterion(predictions, targets)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # NE PAS appeler scheduler.step() ici — CosineAnnealingLR se fait
        # par époque (appelé dans main() après train_one_epoch)

        total["loss"] += loss.item()
        for k in ("obj", "cls", "box"):
            total[k] += loss_dict[k]

        if (batch_idx + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [Ep {epoch}  {batch_idx+1}/{n}]"
                  f"  loss={loss.item():.4f}"
                  f"  obj={loss_dict['obj']:.4f}"
                  f"  cls={loss_dict['cls']:.4f}"
                  f"  box={loss_dict['box']:.4f}"
                  f"  lr={lr:.2e}")

    return {k: v / n for k, v in total.items()}


@torch.no_grad()
def validate(model, loader, criterion, device,
             anchors=ANCHORS, num_classes=NUM_CLASSES,
             compute_map: bool = True) -> dict:
    model.eval()
    total_loss = 0.
    n_batches  = len(loader)
    all_preds, all_targets = [], []

    for spikes, targets in loader:
        spikes = spikes.to(device, non_blocking=True)
        with autocast(enabled=USE_AMP):
            predictions = model(spikes)
            loss, _     = criterion(predictions, targets)
        total_loss += loss.item()

        if compute_map:
            # FIX-1 : décodage corrigé
            dets = decode_predictions(predictions, anchors, conf_thresh=0.05)
            all_preds.extend([d.cpu() for d in dets])
            all_targets.extend([t.cpu() for t in targets])

    # FIX-4 : NMS intégrée dans compute_map50
    map50 = compute_map50(all_preds, all_targets, num_classes) \
            if compute_map else -1.0
    return {"val_loss": total_loss / max(n_batches, 1), "mAP50": map50}


# ===========================================================================
# 7. UTILITAIRES
# ===========================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, epoch: int,
                    metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, path)
    print(f"  ✓ Checkpoint : {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt.get("epoch", 0)
    print(f"  ✓ Checkpoint chargé (époque {epoch})")
    return epoch


# ===========================================================================
# 8. MAIN
# ===========================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SNN Event Camera Detector — VERSION CORRIGÉE")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM       : {vram:.1f} Go")
    print(f"  T_STEPS    : {T_STEPS}")
    print(f"  Résolution : {INPUT_H}×{INPUT_W}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"{'='*60}\n")

    # ── Datasets ─────────────────────────────────────────────────────────
    print("Chargement des datasets...")
    train_ds_raw = EventSpikeDataset(TRAIN_DIR, max_files=MAX_FILES_TRAIN)
    val_ds_raw   = EventSpikeDataset(VAL_DIR,   max_files=MAX_FILES_VAL)

    if len(train_ds_raw) == 0:
        print("\n⚠ Aucune donnée train trouvée. Vérifiez DATA_ROOT.")
        return

    # Optionnel : recalibrer les anchors une seule fois avant l'entraînement
    # Décommentez les deux lignes suivantes la première fois, puis copiez
    # les valeurs dans ANCHORS et recommentez.
    # global ANCHORS
    # ANCHORS = compute_anchors_kmeans(TRAIN_DIR)

    train_ds = CachedDataset(train_ds_raw)
    val_ds   = CachedDataset(val_ds_raw)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    # ── Modèle ────────────────────────────────────────────────────────────
    print("\nConstruction du modèle MSD-SNN...")
    model     = MSDSpikeDetector(num_classes=NUM_CLASSES,
                                  anchors=ANCHORS).to(device)
    criterion = SpikeYOLOLoss(anchors=ANCHORS, num_classes=NUM_CLASSES)
    print(f"  Paramètres : {count_parameters(model):,}")

    # ── AdamW + CosineAnnealingLR + warmup linéaire manuel ──────────────
    # Remplace OneCycleLR qui crashe à la reprise de checkpoint :
    # son total_steps doit être connu AVANT load_checkpoint, ce qui est
    # impossible si le dataset change entre les runs.
    # Le warmup est géré par _warmup_lr() appelé dans train_one_epoch.
    LR_WARMUP_EPOCHS = max(1, int(EPOCHS * 0.10))   # 10% warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_MAX / 25.0,   # démarre bas
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(1, EPOCHS - LR_WARMUP_EPOCHS),
        eta_min = LR_MAX / 1e4,
    )
    scaler = GradScaler(enabled=USE_AMP)

    # ── Reprise ───────────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    resume_path = os.path.join(CHECKPOINT_DIR, "last.pth")
    start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)

    # ── Boucle principale ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_map50    = 0.0
    history       = []

    # Détection du cas où le checkpoint est déjà à la fin
    if start_epoch >= EPOCHS:
        print(f"\n⚠ Le checkpoint last.pth est à l'époque {start_epoch} "
              f"et EPOCHS={EPOCHS}.")
        print(f"  → Supprimez last.pth et best.pth pour repartir de zéro,")
        print(f"    ou augmentez EPOCHS au-delà de {start_epoch}.")
        return

    print(f"\nDébut de l'entraînement — époques {start_epoch+1} → {EPOCHS}\n")
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()

        # Warmup linéaire sur les premières époques (avant CosineAnnealingLR)
        if epoch <= LR_WARMUP_EPOCHS:
            _warmup_lr(optimizer, epoch, LR_WARMUP_EPOCHS, LR_MAX)

        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, epoch,
        )

        # CosineAnnealingLR : step par époque, seulement après le warmup
        if epoch > LR_WARMUP_EPOCHS:
            scheduler.step()

        do_map  = (epoch % MAP_EVAL_EVERY == 0) or (epoch == EPOCHS)
        val_m   = validate(model, val_loader, criterion, device,
                           compute_map=do_map)

        dt     = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        map_str = f"  mAP50={val_m['mAP50']:.4f}" if do_map else "  mAP50=--    "
        print(f"\nÉp {epoch:3d}/{EPOCHS}"
              f"  train={train_m['loss']:.4f}"
              f"  val={val_m['val_loss']:.4f}"
              f"{map_str}"
              f"  lr={lr_now:.2e}"
              f"  {dt:.1f}s")

        record = {**train_m, **val_m,
                  "epoch": epoch, "lr": lr_now, "time_s": dt}
        history.append(record)

        save_checkpoint(model, optimizer, scheduler, epoch, record,
                        os.path.join(CHECKPOINT_DIR, "last.pth"))
        if val_m["val_loss"] < best_val_loss:
            best_val_loss = val_m["val_loss"]
            best_map50    = val_m["mAP50"]
            save_checkpoint(model, optimizer, scheduler, epoch, record,
                            os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"  ★ Best  val={best_val_loss:.4f}  mAP50={best_map50:.4f}")

        print("-" * 60)

    print(f"\nEntraînement terminé.")
    print(f"  Best val_loss : {best_val_loss:.4f}")
    print(f"  Best mAP50    : {best_map50:.4f}")
    _save_report(history, CHECKPOINT_DIR, model,
                 best_val_loss, best_map50, device,
                 LR_WARMUP_EPOCHS)


# ===========================================================================
# 9. RAPPORT FINAL — graphes enrichis + compte-rendu complet
# ===========================================================================

def _save_report(history, out_dir, model, best_val_loss, best_map50, device,
                 lr_warmup_epochs=8):
    """
    Génère dans CHECKPOINT_DIR :
      - training_curves.png    : 6 graphes (loss, composantes, mAP, LR, temps, overfitting)
      - loss_components.png    : zoom sur obj / cls / box séparément
      - training_report.txt    : compte-rendu complet avec diagnostics automatiques
    """
    if not history:
        print("  ⚠ Aucune époque complétée — rapport non généré.")
        print("  Cause probable : start_epoch >= EPOCHS dans le checkpoint.")
        print("  Supprimez last.pth et best.pth puis relancez.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator

    # ── Extraire les séries ───────────────────────────────────────────────
    epochs     = [r["epoch"]    for r in history]
    train_loss = [r["loss"]     for r in history]
    val_loss   = [r["val_loss"] for r in history]
    obj_loss   = [r["obj"]      for r in history]
    cls_loss   = [r["cls"]      for r in history]
    box_loss   = [r["box"]      for r in history]
    lrs        = [r["lr"]       for r in history]
    times      = [r["time_s"]   for r in history]
    gap        = [v - t for t, v in zip(train_loss, val_loss)]

    map_epochs = [r["epoch"] for r in history if r["mAP50"] >= 0]
    map_vals   = [r["mAP50"] for r in history if r["mAP50"] >= 0]

    best_ep    = epochs[val_loss.index(min(val_loss))]
    best_ep_map = map_epochs[map_vals.index(max(map_vals))] if map_vals else best_ep

    # ── Palette ───────────────────────────────────────────────────────────
    C = {
        "train": "#2563EB", "val": "#DC2626", "obj": "#16A34A",
        "cls": "#D97706", "box": "#7C3AED", "map": "#0891B2",
        "lr": "#9333EA", "gap": "#64748B", "best": "#6B7280",
        "time": "#0F766E",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
    })

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Vue d'ensemble (6 sous-graphes)
    # ════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(
        f"SNN Event Camera Detector — Rapport d'entraînement\n"
        f"{len(epochs)} époques  |  best val_loss={best_val_loss:.4f} (ép.{best_ep})  |  "
        f"best mAP@0.5={best_map50:.4f} (ép.{best_ep_map})",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── 1. Loss train / val ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color=C["train"], lw=1.5, marker="o",
             ms=2.5, label="Train loss")
    ax1.plot(epochs, val_loss,   color=C["val"],   lw=1.5, marker="o",
             ms=2.5, label="Val loss")
    ax1.axvline(best_ep, color=C["best"], ls="--", lw=0.8,
                label=f"Best ép.{best_ep}")
    ax1.fill_between(epochs, train_loss, val_loss,
                     alpha=0.07, color=C["gap"])
    ax1.set_title("Loss totale (train vs val)")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 2. Composantes de la loss ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, obj_loss, color=C["obj"], lw=1.2, label="obj (BCE)")
    ax2.plot(epochs, cls_loss, color=C["cls"], lw=1.2, label="cls (BCE)")
    ax2.plot(epochs, box_loss, color=C["box"], lw=1.2, label="box (CIoU)")
    ax2.set_title("Composantes de la loss (train)")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 3. mAP@0.5 ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if map_vals:
        ax3.plot(map_epochs, map_vals, color=C["map"], lw=1.5,
                 marker="D", ms=4, label="mAP@0.5")
        ax3.fill_between(map_epochs, map_vals, alpha=0.12, color=C["map"])
        ax3.axhline(best_map50, color=C["best"], ls="--", lw=0.8,
                    label=f"Best={best_map50:.4f}")
        # Annoter le max
        ax3.annotate(f"{best_map50:.4f}",
                     xy=(best_ep_map, best_map50),
                     xytext=(best_ep_map + max(1, len(epochs)//15),
                             best_map50 + 0.02),
                     fontsize=8, color=C["map"],
                     arrowprops=dict(arrowstyle="->", color=C["map"], lw=0.8))
    else:
        ax3.text(0.5, 0.5, "Pas encore de mAP calculé",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=9, color="gray")
    ax3.set_title("mAP@IoU=0.5")
    ax3.set_ylabel("mAP50")
    ax3.set_ylim(-0.02, 1.05)
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 4. Overfitting gap (val - train) ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    colors_gap = [C["val"] if g > 0.05 else C["obj"] for g in gap]
    ax4.bar(epochs, gap, color=colors_gap, alpha=0.7, width=0.6)
    ax4.axhline(0,    color="black", lw=0.6)
    ax4.axhline(0.05, color=C["val"], ls="--", lw=0.8,
                label="Seuil surapprentissage (0.05)")
    ax4.set_title("Écart val−train (overfitting gap)")
    ax4.set_ylabel("val_loss − train_loss")
    ax4.legend(fontsize=8)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 5. Learning rate ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogy(epochs, lrs, color=C["lr"], lw=1.5, marker="o", ms=2)
    ax5.set_title("Learning rate (échelle log)")
    ax5.set_ylabel("LR")
    ax5.set_xlabel("Époque")
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 6. Temps par époque ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.bar(epochs, [t / 60 for t in times],
            color=C["time"], alpha=0.7, width=0.6)
    ax6.axhline(sum(times) / len(times) / 60, color=C["best"],
                ls="--", lw=0.9, label=f"Moy. {sum(times)/len(times)/60:.1f} min")
    ax6.set_title("Durée par époque")
    ax6.set_ylabel("Temps (min)")
    ax6.set_xlabel("Époque")
    ax6.legend(fontsize=8)
    ax6.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.97)
    fig.savefig(os.path.join(out_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 training_curves.png")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Zoom composantes loss avec moyennes mobiles
    # ════════════════════════════════════════════════════════════════════════
    def moving_avg(arr, w=5):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w)/w, mode="valid").tolist()

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Composantes de la loss — détail + moyenne mobile (w=5)",
                  fontsize=11, fontweight="bold")

    for ax, series, name, color in zip(
        axes2,
        [obj_loss, cls_loss, box_loss],
        ["Objectness (obj BCE)", "Classification (cls BCE)", "Régression boîtes (box CIoU)"],
        [C["obj"], C["cls"], C["box"]],
    ):
        ax.plot(epochs, series, color=color, lw=0.8, alpha=0.4, label="Brut")
        ma = moving_avg(series)
        if len(ma) == len(epochs) - 4:
            ax.plot(epochs[4:], ma, color=color, lw=2.0, label="Moy. mobile w=5")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Époque")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Annoter min
        mn = min(series)
        mn_ep = epochs[series.index(mn)]
        ax.annotate(f"min={mn:.4f}\nép.{mn_ep}",
                    xy=(mn_ep, mn),
                    xytext=(mn_ep + max(1, len(epochs)//10), mn + max(series)*0.05),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loss_components.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  📊 loss_components.png")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — mAP détaillé (si assez de points)
    # ════════════════════════════════════════════════════════════════════════
    if len(map_vals) >= 3:
        fig3, ax3b = plt.subplots(figsize=(10, 4))
        fig3.suptitle("mAP@0.5 — progression détaillée", fontsize=11,
                      fontweight="bold")
        ax3b.plot(map_epochs, map_vals, color=C["map"], lw=1.5,
                  marker="D", ms=5, zorder=3)
        ax3b.fill_between(map_epochs, map_vals, alpha=0.12, color=C["map"])
        for ep, v in zip(map_epochs, map_vals):
            ax3b.annotate(f"{v:.4f}", xy=(ep, v),
                          xytext=(0, 6), textcoords="offset points",
                          ha="center", fontsize=7.5, color=C["map"])
        ax3b.axhline(best_map50, color=C["best"], ls="--", lw=0.8,
                     label=f"Meilleur mAP = {best_map50:.4f} (ép.{best_ep_map})")
        ax3b.set_xlabel("Époque")
        ax3b.set_ylabel("mAP@0.5")
        ax3b.set_ylim(-0.01, max(max(map_vals) * 1.3, 0.1))
        ax3b.legend(fontsize=9)
        ax3b.grid(alpha=0.25)
        plt.tight_layout()
        fig3.savefig(os.path.join(out_dir, "map_detail.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  📊 map_detail.png")

    # ════════════════════════════════════════════════════════════════════════
    # RAPPORT TEXTE ENRICHI
    # ════════════════════════════════════════════════════════════════════════
    rpt = os.path.join(out_dir, "training_report.txt")
    sep  = "=" * 72
    sep2 = "-" * 72

    # Diagnostics automatiques
    def _trend(arr, n=5):
        """Retourne 'baisse', 'stable' ou 'monte' sur les n dernières valeurs."""
        if len(arr) < n + 1:
            return "indéterminé"
        delta = arr[-1] - arr[-n]
        if delta < -0.01:   return "baisse ✓"
        if delta >  0.01:   return "monte ⚠"
        return "stable ~"

    box_plateau = (
        len(box_loss) >= 10 and
        abs(box_loss[-1] - box_loss[-10]) < 0.05
    )
    overfit_now = gap[-1] > 0.05 if gap else False
    map_progressing = (
        len(map_vals) >= 2 and map_vals[-1] > map_vals[0]
    )

    with open(rpt, "w", encoding="utf-8") as f:
        # En-tête
        f.write(f"{sep}\n")
        f.write(f"  SNN EVENT CAMERA DETECTOR — RAPPORT D'ENTRAÎNEMENT\n")
        f.write(f"  Généré le : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{sep}\n\n")

        # Config
        f.write("CONFIGURATION\n" + sep2 + "\n")
        f.write(f"  Résolution        : {INPUT_H}×{INPUT_W} px "
                f"(native {RAW_H}×{RAW_W})\n")
        f.write(f"  T_STEPS           : {T_STEPS} bins temporels\n")
        f.write(f"  DELTA_T           : {DELTA_T_US} µs\n")
        f.write(f"  Batch size        : {BATCH_SIZE}\n")
        f.write(f"  Époques planifiées: {EPOCHS}\n")
        f.write(f"  Époques effectuées: {len(epochs)}\n")
        f.write(f"  LR max            : {LR_MAX}\n")
        f.write(f"  LR warmup         : {lr_warmup_epochs} époques\n")
        f.write(f"  Paramètres        : {count_parameters(model):,}\n")
        f.write(f"  Device            : {device}\n")
        if device.type == "cuda":
            f.write(f"  GPU               : {torch.cuda.get_device_name(0)}\n")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            f.write(f"  VRAM              : {vram:.1f} Go\n")
        f.write(f"  AMP (fp16)        : {USE_AMP}\n\n")

        # Meilleurs résultats
        f.write("MEILLEURS RÉSULTATS\n" + sep2 + "\n")
        f.write(f"  Best val_loss     : {best_val_loss:.6f}  (époque {best_ep})\n")
        f.write(f"  Best mAP@0.5      : {best_map50:.6f}  (époque {best_ep_map})\n")
        f.write(f"  Temps total       : {sum(times)/3600:.2f} h "
                f"({sum(times)/60:.1f} min)\n")
        f.write(f"  Temps/époque moy. : {sum(times)/len(times):.1f} s\n")
        f.write(f"  Temps/époque min  : {min(times):.1f} s\n")
        f.write(f"  Temps/époque max  : {max(times):.1f} s\n\n")

        # Diagnostics automatiques
        f.write("DIAGNOSTICS AUTOMATIQUES\n" + sep2 + "\n")
        f.write(f"  Tendance loss obj (5 dernières ép.) : {_trend(obj_loss)}\n")
        f.write(f"  Tendance loss cls (5 dernières ép.) : {_trend(cls_loss)}\n")
        f.write(f"  Tendance loss box (5 dernières ép.) : {_trend(box_loss)}\n")
        f.write(f"  Tendance val_loss  (5 dernières ép.): {_trend(val_loss)}\n")
        f.write(f"  Plateau box loss (10 ép.)           : "
                f"{'OUI ⚠ → recalibrer anchors' if box_plateau else 'non'}\n")
        f.write(f"  Surapprentissage actuel (gap>0.05)  : "
                f"{'OUI ⚠ → augmenter régularisation' if overfit_now else 'non'}\n")
        f.write(f"  mAP en progression                  : "
                f"{'OUI ✓' if map_progressing else 'non / pas assez de points'}\n")
        f.write(f"  Overfitting gap actuel              : {gap[-1]:+.4f}\n")
        f.write(f"  Overfitting gap max                 : {max(gap):+.4f} "
                f"(ép.{epochs[gap.index(max(gap))]})\n\n")

        # Statistiques loss
        f.write("STATISTIQUES DES COMPOSANTES DE LOSS\n" + sep2 + "\n")
        for name, series in [("obj", obj_loss), ("cls", cls_loss),
                              ("box", box_loss), ("total train", train_loss),
                              ("total val",  val_loss)]:
            arr = np.array(series)
            f.write(f"  {name:<12} : min={arr.min():.4f}  max={arr.max():.4f}"
                    f"  mean={arr.mean():.4f}  std={arr.std():.4f}"
                    f"  last={arr[-1]:.4f}\n")
        f.write("\n")

        # Statistiques mAP
        f.write("STATISTIQUES mAP@0.5\n" + sep2 + "\n")
        if map_vals:
            arr = np.array(map_vals)
            f.write(f"  Points calculés   : {len(map_vals)}"
                    f"  (toutes les {MAP_EVAL_EVERY} époques)\n")
            f.write(f"  mAP min           : {arr.min():.6f}  (ép.{map_epochs[arr.argmin()]})\n")
            f.write(f"  mAP max           : {arr.max():.6f}  (ép.{map_epochs[arr.argmax()]})\n")
            f.write(f"  mAP dernière val. : {arr[-1]:.6f}\n")
            if len(map_vals) >= 2:
                delta = arr[-1] - arr[0]
                f.write(f"  Progression totale: {delta:+.6f}\n")
        else:
            f.write("  Aucune valeur mAP disponible.\n")
        f.write("\n")

        # Tableau par époque
        f.write("HISTORIQUE COMPLET PAR ÉPOQUE\n" + sep2 + "\n")
        header = (f"{'Ép':>4}  {'Train':>8}  {'Val':>8}  {'Gap':>7}  "
                  f"{'obj':>7}  {'cls':>7}  {'box':>7}  "
                  f"{'mAP50':>7}  {'LR':>10}  {'t(s)':>7}\n")
        f.write(header)
        f.write(sep2 + "\n")
        for r in history:
            ep   = r["epoch"]
            g    = r["val_loss"] - r["loss"]
            m    = f"{r['mAP50']:>7.4f}" if r["mAP50"] >= 0 else "     --"
            mark = ""
            if r["val_loss"] == best_val_loss:
                mark += " ★loss"
            if r.get("mAP50", -1) == best_map50 and best_map50 > 0:
                mark += " ★mAP"
            f.write(
                f"{ep:>4}  {r['loss']:>8.4f}  {r['val_loss']:>8.4f}  "
                f"{g:>+7.4f}  {r['obj']:>7.4f}  {r['cls']:>7.4f}  "
                f"{r['box']:>7.4f}  {m}  {r['lr']:>10.2e}  "
                f"{r['time_s']:>7.1f}{mark}\n"
            )
        f.write(sep2 + "\n\n")

        # Fichiers produits
        f.write("FICHIERS PRODUITS\n" + sep2 + "\n")
        for fname in ["best.pth", "last.pth", "training_curves.png",
                      "loss_components.png", "map_detail.png",
                      "training_report.txt"]:
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                size = os.path.getsize(p)
                unit = "Mo" if size > 1e6 else "Ko"
                val  = size / 1e6 if size > 1e6 else size / 1e3
                f.write(f"  {fname:<28} {val:6.1f} {unit}\n")
        f.write(f"\n{sep}\n")

        f.write(f"  Rapport généré automatiquement par _save_report()\n")
        for 
    print(f"  📄 training_report.txt")

if __name__ == "__main__":
    main()