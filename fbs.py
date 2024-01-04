import cv2
from cv2 import ximgproc
import numpy as np

def cal_iou(pred_mask_FBS, gt_mask):
    intersection = np.logical_and(pred_mask_FBS, gt_mask)
    union = np.logical_or(pred_mask_FBS, gt_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    

if __name__ == "__main__":
    mean_iou = 0
    for test_idx in range(1, 61):
        reference = cv2.imread(f'training_dataset/image/{test_idx}.jpg')
        # confidence = cv2.imread(f'output/{test_idx}_clip_confidence.png', cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(f'output/{test_idx}_clip_confidence.png', cv2.IMREAD_GRAYSCALE)
        # target = cv2.imread(f'output/{test_idx}_pred_mask.png', cv2.IMREAD_GRAYSCALE)
        # confidence = np.ones_like(target) * 255
        confidence = target.copy()
        bilatral_solver = ximgproc.createFastBilateralSolverFilter(guide=reference, sigma_spatial=6, sigma_luma=6, sigma_chroma=6)
        # bilatral_solver
        result = bilatral_solver.filter(target, confidence)
        # result[result < 128] = 0
        # result[result >= 128] = 255
        cv2.imwrite(f'output_bilateral/{test_idx}_pred_mask_FBS.jpg', result)
        cv2.imwrite(f'output_bilateral/{test_idx}_pred_mask.jpg', confidence)
        
        # calculate iou
        gt_mask = cv2.imread(f'training_dataset/mask/{test_idx}.jpg', cv2.IMREAD_GRAYSCALE)
        # breakpoint()
        iou = cal_iou(result, gt_mask)
        print(f"{test_idx}.jpg iou_score: {iou:.2f}")
        mean_iou += iou
    print("Finished")
    print(f"iou mean: {mean_iou/60:.2f}")
        
        


