from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")


def segment_image(image_path, mask_path, test_num, prompts=["water"]):
    # load the model and the processor
    
    image = Image.open(image_path)
    mask = np.array(Image.open(mask_path))

    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt").to("cuda")

    # predict
    with torch.no_grad():
        model.to("cuda")
        outputs = model(**inputs)

    # resize the outputs
    preds = nn.functional.interpolate(
        outputs.logits.unsqueeze(0).unsqueeze(0),
        size=(image.size[1], image.size[0]),
        mode="bilinear"
    )
    
    visualize_results(image, preds, prompts, test_num)
    iou_score =  calculate_and_visualize_iou(image, mask, preds, test_num)
    return iou_score


def visualize_results(image, preds, prompts, test_num):
    len_cats = len(prompts)
    fig, ax = plt.subplots(1, len_cats + 1, figsize=(3*(len_cats + 1), 3))
    fig.tight_layout()
    fig.suptitle(f"{test_num}.jpg")
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(image)
    [ax[i+1].imshow(torch.sigmoid(preds[i][0]).detach().cpu()) for i in range(len_cats)]
    [ax[i+1].text(0, -15, category_name) for i, category_name in enumerate(prompts)]
    plt.savefig(f"output/{test_num}_clip_fea.png")
    plt.close()


def calculate_and_visualize_iou(image, mask, preds, test_num):
    threshold = 0.5
    flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))
    # save the CLIP confidence map
    np_preds = flat_preds.view(preds.shape[2], preds.shape[3]).cpu().numpy()
    np_preds = (np_preds * 255).astype(np.uint8)
    image = Image.fromarray(np_preds)
    image.save(f'output/{test_num}_clip_confidence.png')
    
    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
    flat_preds_with_treshold[1:preds.shape[0]+1,:] = flat_preds

    # Get the top mask index for each pixel
    inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))

    # calculate the IoU
    mask_tensor = torch.tensor(mask)
    intersection = torch.logical_and(mask_tensor, inds)
    union = torch.logical_or(mask_tensor, inds)
    iou_score = torch.sum(intersection) / torch.sum(union)
    print(f"{test_num}.jpg iou_score: {iou_score:.2f}")

    visualize_iou(image, mask, inds, test_num, iou_score)

    return iou_score


def visualize_iou(image, mask, inds, test_num, iou_score):
    # breakpoint()
    pred_mask = Image.fromarray((inds.cpu().numpy() * 255).astype(np.uint8))
    pred_mask.save(f'output/{test_num}_pred_mask.png')
    
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(f"iou_score: {iou_score:.2f}")
    fig.tight_layout()
    ax[0].imshow(image)
    ax[0].text(0, -15, f"{test_num}.jpg")
    ax[0].axis('off')
    ax[1].imshow(mask)
    ax[1].text(0, -15, "mask")
    ax[1].axis('off')
    ax[2].imshow(inds)
    ax[2].text(0, -15, "predicted mask")
    ax[2].axis('off')
    plt.savefig(f"output/{test_num}_clip_mask.png")
    plt.close()

if __name__ == "__main__":
    import time
    start_time = time.time()
    iou_list = []
    for test_num in range(1, 61):
      image_path = f"training_dataset/image/{test_num}.jpg"
      mask_path = f"training_dataset/mask/{test_num}.jpg"
      prompts = ["water"]
      iou = segment_image(image_path, mask_path, test_num, prompts)
      iou_list.append(iou)
    
    print(f"iou mean: {np.mean(iou_list):.2f}")
    print(f"total time: {time.time() - start_time:.2f}")
