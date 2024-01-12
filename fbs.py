import cv2
from cv2 import ximgproc
import numpy as np
import argparse
import os

def cal_iou(pred_mask_FBS, gt_mask):
    intersection = np.logical_and(pred_mask_FBS, gt_mask)
    union = np.logical_or(pred_mask_FBS, gt_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def fbs(reference_path, target_dir, confidence_dir, test_idx, output_dir="demo_output"):
    # read reference, target, confidence
    reference = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    target_path = os.path.join(target_dir, f'{test_idx}_clip_confidence.png')
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    confidence = target.copy()
    # create bilateral solver
    bilatral_solver = ximgproc.createFastBilateralSolverFilter(guide=reference, sigma_spatial=6, sigma_luma=6, sigma_chroma=6)
    result = bilatral_solver.filter(target, confidence)
    # thresholding
    threshold = np.max(confidence) - confidence
    result[result < threshold] = 0
    result[result >= threshold] = 255
    # write output
    cv2.imwrite(f'{output_dir}/output{test_idx}.jpg', result)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--thresholding', type=str, default='binary', choices=['binary', 'otsu', 'confidence', 'none'], help='thresholding method')
    args = arg_parser.parse_args()
    
    mean_iou = 0
    for test_idx in range(1, 61):
        reference = cv2.imread(f'training_dataset/image/{test_idx}.jpg')
        confidence = cv2.imread(f'output/{test_idx}_clip_confidence.png', cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(f'output/{test_idx}_pred_mask.png', cv2.IMREAD_GRAYSCALE)
        # target = confidence.copy()
        bilatral_solver = ximgproc.createFastBilateralSolverFilter(guide=reference, 
                                                                   sigma_spatial=6, 
                                                                   sigma_luma=6, 
                                                                   sigma_chroma=6)
        result = bilatral_solver.filter(target, confidence)
        
        if args.thresholding == 'binary':
            ## Binary thresholding
            threshold = 128
            result[result < threshold] = 0
            result[result >= threshold] = 255
        elif args.thresholding == 'otsu':
            ## Otsu thresholding
            ret2, th2 = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = th2
        elif args.thresholding == 'confidence':
            threshold = np.max(confidence) - confidence
            result[result < threshold] = 0
            result[result >= threshold] = 255
        elif args.thresholding == 'none':
            pass
        
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
        
        


