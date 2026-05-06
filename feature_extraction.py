import argparse
import os
from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import time

# =========================
# Folder & listing
# =========================
import json

def parse_aug_map(arg_str):
    if not arg_str:
        return None
    try:
        m = json.loads(arg_str)
        if not isinstance(m, dict):
            raise ValueError("aug-map must be a JSON object")
        # normalize keys to strings, values to nonnegative ints
        out = {}
        for k,v in m.items():
            out[str(k)] = int(v)
            if out[str(k)] < 0:
                out[str(k)] = 0
        return out
    except Exception as e:
        raise ValueError(f"--aug-map parse error: {e}")


def list_images(root: Path):
    root = Path(root)
    out = []
    for cls in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img in sorted(cls.rglob('*')):
            if img.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
                out.append((cls.name, img))
    return out

# =========================
# Deep backbone (frozen)
# =========================

class ResNetFeature(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        if backbone == 'resnet18':
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == 'resnet50':
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError('Unsupported backbone')
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.feat_dim = feat_dim
    def forward(self, x):
        with torch.no_grad():
            z = self.backbone(x)
        return z

PREPROC = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_deep_features_from_pils(pil_images, backbone='resnet18', device='cpu', batch=32):
    model = ResNetFeature(backbone).to(device).eval()
    feats = []
    for i in range(0, len(pil_images), batch):
        batch_imgs = pil_images[i:i+batch]
        x = [PREPROC(im) for im in batch_imgs]
        x = torch.stack(x).to(device)
        with torch.no_grad():
            z = model(x)  # [B, feat_dim]
        feats.append(z.cpu().numpy())
    return np.vstack(feats)

# =========================
# Classic features (simple)
# =========================

import cv2

def classic_from_pil(im: Image.Image):
    img = np.array(im.convert('RGB'))
    R,G,B = img[:,:,0].astype(np.float32), img[:,:,1].astype(np.float32), img[:,:,2].astype(np.float32)
    feats = {}
    feats['R_mean'], feats['G_mean'], feats['B_mean'] = R.mean(), G.mean(), B.mean()
    feats['R_std'],  feats['G_std'],  feats['B_std']  = R.std(),  G.std(),  B.std()
    eps=1e-6
    feats['RG_ratio'] = feats['R_mean']/(feats['G_mean']+eps)
    total = feats['R_mean']+feats['G_mean']+feats['B_mean']+eps
    feats['Red_norm'] = feats['R_mean']/total
    feats['Excess_red'] = feats['R_mean'] - 0.5*(feats['G_mean']+feats['B_mean'])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0]
    h_hist,_ = np.histogram(H, bins=12, range=(0,180), density=True)
    for i,v in enumerate(h_hist):
        feats[f'H_hist_{i:02d}']=float(v)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    feats['lap_var'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return feats

# =========================
# Augmentation (TRAIN only)
# =========================

def augment_pil(img: Image.Image, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # H-flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # small rotation
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR)
    # random resized crop (keep 80–100% area)
    w, h = img.size
    scale = random.uniform(0.80, 1.00)
    new_w, new_h = int(w*scale), int(h*scale)
    if new_w < w and new_h < h:
        x0 = random.randint(0, w - new_w)
        y0 = random.randint(0, h - new_h)
        img = img.crop((x0, y0, x0+new_w, y0+new_h)).resize((w, h), Image.BILINEAR)
    # light color jitter
    def jitter(enh_cls, rng):
        f = random.uniform(*rng)
        return enh_cls(img).enhance(f)
    img = jitter(ImageEnhance.Brightness, (0.85, 1.15))
    img = jitter(ImageEnhance.Contrast,   (0.85, 1.15))
    img = jitter(ImageEnhance.Color,      (0.85, 1.15))
    return img

# =========================
# MI & selection helpers
# =========================

FEATURE_EXCLUDE = {"label","image_path"}

def compute_mi(df):
    X = df[[c for c in df.columns if c not in FEATURE_EXCLUDE]].values
    y,_ = pd.factorize(df['label'])
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    return pd.DataFrame({'feature':[c for c in df.columns if c not in FEATURE_EXCLUDE], 'MI':mi}).sort_values('MI', ascending=False)

def compute_selected_features(mi_df, df_train, topk):
    if topk and mi_df is not None and len(mi_df)>0:
        ranked = [f for f in mi_df['feature'].tolist() if f in df_train.columns and f not in FEATURE_EXCLUDE]
        return ranked[:topk]
    return [c for c in df_train.columns if c not in FEATURE_EXCLUDE]


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--test-dir', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--features', choices=['deep','classic','hybrid'], default='deep')
    ap.add_argument('--structure', choices=['naive','chowliu','tan','k2'], default='naive')
    ap.add_argument('--deep-backbone', choices=['resnet18','resnet50'], default='resnet18')
    ap.add_argument('--deep-dim', type=int, default=128)
    ap.add_argument('--pca-whiten', action='store_true')
    ap.add_argument('--topk', type=int, default=0)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--max-parents', type=int, default=2)
    ap.add_argument('--uniform-prior', action='store_true', help='Use uniform class priors instead of empirical frequencies')
    ap.add_argument('--aug', type=int, default=0, help='Number of augmented copies per TRAIN image (0=no augmentation)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--mi', action='store_true')
    ap.add_argument(
    '--aug-map', type=str, default=None,
    help='JSON dict of per-class augmentation copies, e.g. {"Hypersensitivity":5,"Fungal_infections":4,"default":2}. Overrides --aug if provided.'
)
    ap.add_argument('--seed', type=int, default=42,
    help='Random seed for reproducibility (controls augmentations and other stochastic ops)')

    args = ap.parse_args()
    
    t = time.time()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load lists
    train_list = list_images(Path(args.data))
    test_list  = list_images(Path(args.test_dir))

    # -------- Base DataFrames
    df_tr = pd.DataFrame({'label':[c for c,_ in train_list], 'image_path':[str(p) for _,p in train_list]})
    df_te = pd.DataFrame({'label':[c for c,_ in test_list],  'image_path':[str(p) for _,p in test_list]})

    # -------- Build TRAIN/TEST PIL lists (apply augmentation to TRAIN only)
    # train_pils, train_labels, train_paths = [], [], []
    # for label, path in zip(df_tr['label'].tolist(), df_tr['image_path'].tolist()):
    #     img0 = Image.open(path).convert('RGB')
    #     # original
    #     train_pils.append(img0)
    #     train_labels.append(label)
    #     train_paths.append(path)
    #     # augmented copies
    #     for k in range(args.aug):
    #         train_pils.append(augment_pil(img0))
    #         train_labels.append(label)
    #         train_paths.append(path + f'#aug{k+1}')
    # df_tr = pd.DataFrame({'label': train_labels, 'image_path': train_paths})

    # -------- Build TRAIN/TEST PIL lists (apply augmentation to TRAIN only)
    aug_map = parse_aug_map(args.aug_map)
    train_pils, train_labels, train_paths = [], [], []

    for label, path in zip(df_tr['label'].tolist(), df_tr['image_path'].tolist()):
        img0 = Image.open(path).convert('RGB')

        # Always add the original
        train_pils.append(img0)
        train_labels.append(label)
        train_paths.append(path)

        # Decide how many extra copies to add
        if aug_map is not None:
            extra = aug_map.get(label, aug_map.get("default", 0))
        else:
            extra = args.aug  # global augmentation

        # Add class-targeted augmented copies
        for k in range(int(extra)):
            train_pils.append(augment_pil(img0, args.seed))
            train_labels.append(label)
            train_paths.append(f"{path}#aug{k+1}")

    # Rebuild TRAIN df with expanded rows
    df_tr = pd.DataFrame({'label': train_labels, 'image_path': train_paths})


    test_pils = [Image.open(p).convert('RGB') for p in df_te['image_path'].tolist()]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -------- Deep features
    if args.features in ('deep','hybrid'):
        print('[1/6] Extracting deep features (TRAIN, aug if any)...')
        Z_tr = extract_deep_features_from_pils(train_pils, backbone=args.deep_backbone, device=device)
        # np.save("array.npy", Z_tr)      # Save
        #Z_tr = np.load("array.npy", allow_pickle=True)  # Load
        print('[1/6] Extracting deep features (TEST)...')
        Z_te = extract_deep_features_from_pils(test_pils,  backbone=args.deep_backbone, device=device)
        print('[1/6] PCA on deep (Train only)...')
        pca = PCA(n_components=args.deep_dim, whiten=args.pca_whiten, random_state=42)
        Z_tr_pca = pca.fit_transform(Z_tr)
        Z_te_pca = pca.transform(Z_te)
        for i in range(Z_tr_pca.shape[1]):
            df_tr[f'deep_{i+1:03d}']=Z_tr_pca[:,i]
            df_te[f'deep_{i+1:03d}']=Z_te_pca[:,i]

    # -------- Classic features
    if args.features in ('classic','hybrid'):
        print('[1/6] Extracting classic features (TRAIN, same augmented PILs)...')
        rows_tr=[classic_from_pil(im) for im in train_pils]
        print('[1/6] Extracting classic features (TEST)...')
        rows_te=[classic_from_pil(im) for im in test_pils]
        df_tr = pd.concat([df_tr, pd.DataFrame(rows_tr)], axis=1)
        df_te = pd.concat([df_te, pd.DataFrame(rows_te)], axis=1)

    # -------- MI & Top-K (TRAIN only)
    mi_csv = out_dir/ 'mi_train.csv'
    mi_df=None
    if args.features in ('classic','hybrid','deep'):
        if args.mi or args.all or not mi_csv.exists():
            print('[2/6] MI on TRAIN features...')
            mi_df = compute_mi(df_tr)
            mi_df.to_csv(mi_csv, index=False)
        else:
            mi_df = pd.read_csv(mi_csv)
        selected_feats = compute_selected_features(mi_df, df_tr, args.topk)
        df_tr = df_tr[selected_feats + ['label','image_path']]
        df_te = df_te.reindex(columns=selected_feats + ['label','image_path'])
        print(f'    -> Using {len(selected_feats)} features (Top-{args.topk} if >0).')
    else:
        selected_feats = [c for c in df_tr.columns if c not in FEATURE_EXCLUDE]

    df_tr.to_csv(out_dir / f"train_tabular_{args.seed}.csv", index=False)

    df_te.to_csv(out_dir / f"test_tabular_{args.seed}.csv", index=False)
    
    print(time.time()-t)
    
    print('Data saved.')

if __name__=='__main__':
    main()
