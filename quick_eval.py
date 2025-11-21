import numpy as np
import cv2
import time
import os
import argparse
from prettytable import PrettyTable
from src.cmeasure import GeneralContextMeasure, CamoContextMeasure, ContextMeasure
from src.metrics import compute_all_metrics

def evaluate(img, gt, fm):
    # two usage
    # first:
    GeneralCmeasure = GeneralContextMeasure()
    CamoCmeasure = CamoContextMeasure()
    gcm = GeneralCmeasure.compute(fm,gt)
    ccm = CamoCmeasure.compute(fm,gt,img)

    # second:
    # CMeasure = ContextMeasure()
    # cm = CMeasure.compute(fm,gt,img)

    # other saliency-era metrics:
    others = compute_all_metrics(fm, gt)

    # ---- PrettyTable 输出 ----
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    # First part: context measures
    table.add_row(["General Context-measure", f"{gcm:.8f}"])
    table.add_row(["Camouflaged Context-measure", f"{ccm:.8f}"])

    # Divider row
    table.add_row(["-" * 25, "-" * 15])

    # Other saliency-era metrics
    for metric, value in others.items():
        table.add_row([metric, f"{value:.8f}"])

    # Print table
    print(table)


def visualize(img, gt):
    save_dir = f'vis_cd'
    os.makedirs(save_dir, exist_ok=True)
    cm = ContextMeasure()
    img_recon, cd = cm.visualize(img, gt)
    local_cd = cv2.GaussianBlur(cd,ksize=(5,5),sigmaX=5,sigmaY=5)
    heatmap = cv2.applyColorMap((local_cd*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = 0.7*heatmap + 0.3*np.float32(img)
    cv2.imwrite(os.path.join(save_dir, f'heatmap.png'), cam)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    index = 465
    img = cv2.imread(f'img/{index}.jpg')
    gt = cv2.imread(f'img/{index}_gt.png', cv2.IMREAD_GRAYSCALE)
    fm = cv2.imread(f'img/{index}_fm.png', cv2.IMREAD_GRAYSCALE)

    evaluate(img, gt, fm)  # quick evaluation
    visualize(img, gt)  # visualize camouflage degree


