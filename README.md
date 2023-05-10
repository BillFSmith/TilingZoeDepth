# Tiling ZoeDepth outputs for higher resolution

This is an adapted version of https://colab.research.google.com/github/isl-org/ZoeDepth/blob/main/notebooks/ZoeDepth_quickstart.ipynb

Corresponding paper : [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288v1)


Here, higher resolution depth maps are generated from the following process:
    Generate a depth map for the overall image
    Split original image into overlapping tiles
    Generate depth maps for the tiles
    Reassemble into a single depth map by applying gradient masks and average weighting from first depth map
    Repeat steps 2-4 at higher resolution
    Combine all three depth maps by:
        Calculate edge filter from original RGB image
        Blur edge filter and use as mask for high resolution depth map
        Apply masked high resolution to average of low and medium resolution depth maps

Colab here: https://colab.research.google.com/drive/1taTL_8GVx1G93ZXp_o-s4FLL-SY6N8TC?usp=sharing
