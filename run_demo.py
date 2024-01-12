from dip_final import segment_image
from fbs import fbs

if __name__ == "__main__":
    demo_input_dir = "demo_input/"
    demo_output_dir = "demo_output/"
    target_dir = "output/"
    for test_num in range(1, 13):
        image_path = demo_input_dir + f"input{test_num}.jpg"
        mask_path = None
        prompts = ["water"]
        segment_image(image_path, mask_path, test_num, prompts, demo_output_dir)    
        fbs(image_path, target_dir=target_dir, confidence_dir=target_dir, test_idx=test_num, output_dir=demo_output_dir)