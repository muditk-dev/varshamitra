"""VarshaMitra Regime Classifier Module.
=======================================
Implements two machine learning classifiers for active monsoon regime identification:
1. XGBoost Tabular Baseline Classifier (per-cell diagnostic features)
2. PyTorch 2D Spatial U-Net Classifier (capturing spatial context across Maharashtra)

Evaluates on held-out temporal split and produces confusion matrices.
"""

import logging
from typing import Dict, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("varshamitra.classifier")

FEATURE_COLS = [
    "precip_raw", "elevation", "slope_lon", "slope_lat",
    "u850", "v850", "u200", "v200", "mslp", "rh850",
    "wind_shear", "wind_speed_850", "upslope_flow",
    "vorticity", "mslp_anomaly", "q_850", "mfc",
    "precip_roll3", "coastal_proximity"
]

NUM_REGIMES = 6


class XGBoostRegimeClassifier:
    """Tabular Gradient Boosting baseline classifier for monsoon regimes."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=NUM_REGIMES,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4
        )
        self.feature_names = FEATURE_COLS
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info("Training XGBoost baseline regime classifier...")
        # Ensure all NUM_REGIMES classes are represented so XGBoost multi:softprob initializes all 6 classes
        missing_classes = set(range(NUM_REGIMES)) - set(np.unique(y))
        if missing_classes:
            logger.info(f"Adding placeholder exemplars for rare unobserved regimes: {missing_classes}")
            X_dummy = np.zeros((len(missing_classes), X.shape[1]), dtype=X.dtype)
            y_dummy = np.array(list(missing_classes), dtype=y.dtype)
            weights_real = np.ones(len(y), dtype=np.float32)
            weights_dummy = np.full(len(missing_classes), 1e-5, dtype=np.float32)
            X = np.vstack([X, X_dummy])
            y = np.concatenate([y, y_dummy])
            sample_weight = np.concatenate([weights_real, weights_dummy])
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        logger.info("XGBoost training complete.")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class SpatialConvNetClassifier(nn.Module):
    """Convolutional Neural Network capturing 2D spatial meteorological patterns.
    Takes 2D feature maps over Maharashtra (channels = num_features, H = n_lats, W = n_lons)
    and predicts per-cell regime logits (channels = NUM_REGIMES, H, W).
    """
    
    def __init__(self, in_channels: int = len(FEATURE_COLS), num_classes: int = NUM_REGIMES):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Decoder / Classifier Head (preserves full resolution)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> logits: (B, num_classes, H, W)
        feat1 = self.enc1(x)
        feat2 = self.enc2(feat1)
        out = self.out_conv(feat2)
        return out


class SpatialGridDataset(Dataset):
    """PyTorch Dataset yielding 2D spatial maps for each forecast day."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # features: (T, C, H, W), labels: (T, H, W)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_and_evaluate_regime_classifiers(
    featured_ds: xr.Dataset,
    train_split_ratio: float = 0.75,
    epochs: int = 15
) -> Dict[str, any]:
    """Train both XGBoost and ConvNet on temporal split and return comprehensive evaluation metrics."""
    logger.info("Preparing train/test split for regime classifier evaluation...")
    
    times = featured_ds["time"].values
    n_times = len(times)
    split_idx = int(n_times * train_split_ratio)
    
    # Extract 4D feature array: (T, C, H, W)
    feature_list = []
    for f in FEATURE_COLS:
        arr = featured_ds[f].values
        # Normalize feature per-variable across training set
        f_mean = np.mean(arr[:split_idx])
        f_std = np.std(arr[:split_idx]) + 1e-6
        normed = (arr - f_mean) / f_std
        feature_list.append(normed)
    X_4d = np.stack(feature_list, axis=1).astype(np.float32)  # (T, C, H, W)
    y_3d = featured_ds["regime"].values.astype(np.int64)      # (T, H, W)
    
    # Train / Test split along temporal axis
    X_train_4d, X_test_4d = X_4d[:split_idx], X_4d[split_idx:]
    y_train_3d, y_test_3d = y_3d[:split_idx], y_3d[split_idx:]
    
    # Tabular Flattening for XGBoost
    T_tr, C, H, W = X_train_4d.shape
    T_te = X_test_4d.shape[0]
    
    X_train_tab = X_train_4d.transpose(0, 2, 3, 1).reshape(-1, C)
    y_train_tab = y_train_3d.reshape(-1)
    X_test_tab = X_test_4d.transpose(0, 2, 3, 1).reshape(-1, C)
    y_test_tab = y_test_3d.reshape(-1)
    
    # 1. Train XGBoost Baseline
    xgb_clf = XGBoostRegimeClassifier()
    xgb_clf.fit(X_train_tab, y_train_tab)
    y_pred_xgb = xgb_clf.predict(X_test_tab)
    
    xgb_acc = float(accuracy_score(y_test_tab, y_pred_xgb))
    xgb_cm = confusion_matrix(y_test_tab, y_pred_xgb, labels=list(range(NUM_REGIMES)))
    xgb_report = classification_report(y_test_tab, y_pred_xgb, output_dict=True, zero_division=0)
    logger.info(f"XGBoost Baseline Test Accuracy: {xgb_acc * 100:.2f}%")
    
    # 2. Train Spatial ConvNet
    logger.info("Training Spatial ConvNet (U-Net style) in PyTorch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialConvNetClassifier(in_channels=C, num_classes=NUM_REGIMES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    train_dataset = SpatialGridDataset(X_train_4d, y_train_3d)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)  # (B, 6, H, W)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(f"  Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss / len(train_dataset):.4f}")
            
    # Evaluate ConvNet
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.tensor(X_test_4d, dtype=torch.float32).to(device)
        test_logits = model(test_x_tensor)  # (T_te, 6, H, W)
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()  # (T_te, H, W)
        
    y_pred_cnn = test_preds.reshape(-1)
    cnn_acc = float(accuracy_score(y_test_tab, y_pred_cnn))
    cnn_cm = confusion_matrix(y_test_tab, y_pred_cnn, labels=list(range(NUM_REGIMES)))
    cnn_report = classification_report(y_test_tab, y_pred_cnn, output_dict=True, zero_division=0)
    logger.info(f"Spatial ConvNet Test Accuracy: {cnn_acc * 100:.2f}%")
    
    return {
        "xgb_model": xgb_clf,
        "cnn_model": model,
        "xgb_accuracy": xgb_acc,
        "cnn_accuracy": cnn_acc,
        "xgb_confusion_matrix": xgb_cm,
        "cnn_confusion_matrix": cnn_cm,
        "xgb_report": xgb_report,
        "cnn_report": cnn_report,
        "split_idx": split_idx
    }
