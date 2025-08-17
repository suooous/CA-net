# %%
import os
import numpy as np
from glob import glob
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import AoPDataset
import matplotlib.pyplot as plt
import torch
from loss import DiceCoefficient, DiceLoss
import torch.optim as optim
from upernet import UPerNet
from PIL import Image
import torch.nn.functional as F
import cv2

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from torch.utils.data import DataLoader
from dataset import AoPDataset
from upernet import UPerNet

# —— helper functions —— 

def fit_ellipse_params(mask, cls):
    bin_mask = (mask == cls).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    return cv2.fitEllipse(cnt)  # ((cx,cy),(MA,ma),angle_deg)

def ellipse_point(ellipse, t):
    (cx,cy),(MA,ma),angle = ellipse
    a, b      = MA/2, ma/2
    th        = np.deg2rad(angle)
    x = cx +  a*np.cos(t)*np.cos(th) - b*np.sin(t)*np.sin(th)
    y = cy +  a*np.cos(t)*np.sin(th) + b*np.sin(t)*np.cos(th)
    return np.array([x,y])

def ellipse_normal(ellipse, t):
    (cx,cy),(MA,ma),angle = ellipse
    a, b      = MA/2, ma/2
    th        = np.deg2rad(angle)
    nx =  a*np.cos(t)*np.cos(th) - b*np.sin(t)*np.sin(th)
    ny =  a*np.cos(t)*np.sin(th) + b*np.sin(t)*np.cos(th)
    return np.array([nx, ny])

def ellipse_tangent_dir(ellipse, t):
    (cx,cy),(MA,ma),angle = ellipse
    a, b      = MA/2, ma/2
    th        = np.deg2rad(angle)
    tx = -a*np.sin(t)*np.cos(th) - b*np.cos(t)*np.sin(th)
    ty = -a*np.sin(t)*np.sin(th) + b*np.cos(t)*np.cos(th)
    return np.array([tx, ty])

def get_major_axis_endpoints(ellipse):
    """
    Given ellipse = ((cx,cy),(w,h),angle_deg),
    returns the two endpoints of the true major axis.
    """
    (cx, cy), (w, h), angle = ellipse

    # determine which is the major diameter
    if w >= h:
        a = w / 2.0
        theta = np.deg2rad(angle)
    else:
        a = h / 2.0
        theta = np.deg2rad(angle + 90.0)

    dx =  a * np.cos(theta)
    dy =  a * np.sin(theta)

    P1 = np.array([cx + dx, cy + dy])
    P2 = np.array([cx - dx, cy - dy])
    return P1, P2

def get_major_axis_right_endpoint(ellipse):
    """
    Returns the endpoint of the major axis with the larger x-coordinate.
    """
    P1, P2 = get_major_axis_endpoints(ellipse)
    return P1 if P1[0] > P2[0] else P2


def get_tangent_point(P, ellipse, n_samples=3600):
    """
    From an external point P, find the two candidate tangent points on the ellipse
    and return the right-hand one (with larger x-coordinate).

    Args:
        P:     array-like, shape (2,)
        ellipse: tuple as returned by cv2.fitEllipse
        n_samples: how many t-values to discretize [0,2π)

    Returns:
        Q: the chosen contact point on the ellipse, shape (2,)
    """
    # 1) sample parameters
    ts = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    # 2) compute residual = |(P−X(t))·N(t)| for each t
    residuals = [
        (abs((P - ellipse_point(ellipse, t)) @ ellipse_normal(ellipse, t)), t)
        for t in ts
    ]
    # 3) pick two best candidate ts
    residuals.sort(key=lambda x: x[0])
    t0, t1 = residuals[0][1], residuals[1][1]
    # 4) compute their points
    Q0 = ellipse_point(ellipse, t0)
    Q1 = ellipse_point(ellipse, t1)
    # 5) choose the right‐hand one
    return Q0 if Q0[0] >= Q1[0] else Q1

# %% set up data & model

data_root = r'D:\data\dataset'
test_img_files = [os.path.join(data_root, 'test_image', f) for f in os.listdir(os.path.join(data_root, 'test_image'))]
test_lbl_files = [os.path.join(data_root, 'test_label', f) for f in os.listdir(os.path.join(data_root, 'test_label'))]

test_dataset = AoPDataset(img_paths=test_img_files, lbl_paths=test_lbl_files)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UPerNet(in_channel=1, num_class=3).to(device)
model.load_state_dict(torch.load("best_model_upernet.pth"))
model.eval()

out_dir = r"predictions_ellipses"
os.makedirs(out_dir, exist_ok=True)

results = []
with torch.no_grad():
    for idx, (imgs, lbls) in enumerate(test_loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)['fuse']
        pred_lbl = torch.argmax(preds, dim=1).cpu().squeeze().numpy().astype(np.uint8)
        img_np   = imgs.cpu().squeeze().numpy()
        lbl_np   = lbls.cpu().squeeze().numpy().astype(np.uint8)
        H, W     = pred_lbl.shape

        # prepare gray canvases
        lbl_cm  = np.zeros((H, W, 3), dtype=np.uint8)
        pred_cm = np.zeros((H, W, 3), dtype=np.uint8)
        lbl_cm[lbl_np==1]   = [220,220,220]; lbl_cm[lbl_np==2]   = [140,140,140]
        pred_cm[pred_lbl==1] = [200,200,200]; pred_cm[pred_lbl==2] = [100,100,100]

        # fit ellipses
        gt_e1   = fit_ellipse_params(lbl_np,   1)
        gt_e2   = fit_ellipse_params(lbl_np,   2)
        pr_e1   = fit_ellipse_params(pred_lbl, 1)
        pr_e2   = fit_ellipse_params(pred_lbl, 2)

        # draw ellipse outlines (class colors)
        for ellipse, canvas, cls in [(gt_e1, lbl_cm,   1),
                                     (gt_e2, lbl_cm,   2),
                                     (pr_e1, pred_cm,  1),
                                     (pr_e2, pred_cm,  2)]:
            if ellipse is None: continue
            color = (0,255,0) if cls==1 else (255,0,0)
            cv2.ellipse(canvas, ellipse, color=color, thickness=2)

        if gt_e1 is not None:
            P_gt_major = get_major_axis_right_endpoint(gt_e1)
            cv2.circle(lbl_cm,
                       tuple(P_gt_major.astype(int)),
                       radius=5,
                       color=(0,255,255),  # yellow
                       thickness=-1)

        if pr_e1 is not None:
            P_pr_major = get_major_axis_right_endpoint(pr_e1)
            cv2.circle(pred_cm,
                       tuple(P_pr_major.astype(int)),
                       radius=5,
                       color=(0,255,255),  # yellow
                       thickness=-1)

        # ==== GROUND TRUTH ====  
        if gt_e1 is not None and gt_e2 is not None:
            # 1) major‐axis endpoints  
            M1, M2 = get_major_axis_endpoints(gt_e1)  
            p1i, p2i = tuple(M1.astype(int)), tuple(M2.astype(int))
            # draw major axis in yellow  
            cv2.line(lbl_cm, p1i, p2i, color=(0,255,255), thickness=2)

            # 2) P = right‐end of major axis, Q = tangent contact on cls2  
            P = get_major_axis_right_endpoint(gt_e1)  
            Q = get_tangent_point(P, gt_e2)

            # 3) draw tangent line in yellow  
            V = Q - P
            V /= np.linalg.norm(V)
            L = max(H, W)*2
            t1 = (P +  V*L).astype(int)
            t2 = (P -  V*L).astype(int)
            cv2.line(lbl_cm, tuple(t1), tuple(t2), color=(0,255,255), thickness=2)

            # 4) mark contact Q in white  
            cv2.circle(lbl_cm, tuple(Q.astype(int)), radius=5, color=(255,255,255), thickness=-1)

            # 5) compute & show angle at P  
            axis_vec = (M2 - M1)
            axis_vec /= np.linalg.norm(axis_vec)
            ang = 180 - np.degrees(np.arccos(np.clip(np.dot(axis_vec, V), -1.0, 1.0)))
            cv2.putText(
                lbl_cm,
                f"{ang:.1f}",
                org=(int(P[0]+10), int(P[1]-10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0,255,255),
                thickness=2
            )

        # ==== PREDICTION ====  
        if pr_e1 is not None and pr_e2 is not None:
            M1p, M2p = get_major_axis_endpoints(pr_e1)
            p1ip, p2ip = tuple(M1p.astype(int)), tuple(M2p.astype(int))
            cv2.line(pred_cm, p1ip, p2ip, color=(0,255,255), thickness=2)

            Pp = get_major_axis_right_endpoint(pr_e1)
            Qp = get_tangent_point(Pp, pr_e2)

            Vp = Qp - Pp
            Vp /= np.linalg.norm(Vp)
            Lp = max(H, W)*2
            tp1 = (Pp +  Vp*Lp).astype(int)
            tp2 = (Pp -  Vp*Lp).astype(int)
            cv2.line(pred_cm, tuple(tp1), tuple(tp2), color=(0,255,255), thickness=2)

            cv2.circle(pred_cm, tuple(Qp.astype(int)), radius=5, color=(255,255,255), thickness=-1)

            axis_vec_p = (M2p - M1p)
            axis_vec_p /= np.linalg.norm(axis_vec_p)
            ang_p = 180 - np.degrees(np.arccos(np.clip(np.dot(axis_vec_p, Vp), -1.0, 1.0)))
            cv2.putText(
                pred_cm,
                f"{ang_p:.1f}",
                org=(int(Pp[0]+10), int(Pp[1]-10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0,255,255),
                thickness=2
            )

        results.append({
            "sample": idx,
            "gt_angle": ang,
            "pred_angle": ang_p
        })

        # … now plot & save as before …
        fig, ax = plt.subplots(1,3,figsize=(12,4))
        ax[0].imshow(img_np, cmap='gray'); ax[0].axis('off'); ax[0].set_title("Input")
        ax[1].imshow(lbl_cm);               ax[1].axis('off'); ax[1].set_title("GT")
        ax[2].imshow(pred_cm);              ax[2].axis('off'); ax[2].set_title("Pred")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{idx:03d}.jpg"), dpi=150)
        plt.close(fig)

# <-- new: save to CSV -->
df = pd.DataFrame(results)
df.to_csv("angles.csv", index=False)

# %%