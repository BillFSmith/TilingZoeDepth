# Tiling ZoeDepth outputs for higher resolution
v4 is experimental and uses Lotus Depth

https://colab.research.google.com/drive/1C9gAdMezh3b0O_yXuQpIunSasoP1BzSz

v3 has a more reliable upload system for larger files. Can take multiple files at once:

https://colab.research.google.com/drive/1Wi-1Ji_fhcoGpK-drT4dVrl5AjfVUQ5M

v2 has a GUI and STL generation:

https://colab.research.google.com/drive/1wbbXpMC_UUwE3e7Tifq9fYNnd5Rn0zna

v1 is broken into sections:

https://colab.research.google.com/drive/1taTL_8GVx1G93ZXp_o-s4FLL-SY6N8TC

This is an adapted version of https://colab.research.google.com/github/isl-org/ZoeDepth/blob/main/notebooks/ZoeDepth_quickstart.ipynb

Corresponding paper : [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288v1)

Here, higher resolution depth maps are generated from the following process:

1)  Generate a depth map for the overall image    
2)  Split original image into overlapping tiles    
3)  Generate depth maps for the tiles    
4)  Reassemble into a single depth map by applying gradient masks and average weighting from first depth map    
5)  Repeat steps 2-4 at higher resolution
6)  Combine all three depth maps by: <br>
        a) Calculate edge filter from original RGB image<br>
        b) Blur edge filter and use as mask for high resolution depth map<br>
        c) Apply masked high resolution to average of low and medium resolution depth maps

The difference between the low resolution original and the new version can be seen below:    
![zoe_depth_map_16bit_low(4)](https://github.com/BillFSmith/TilingZoeDepth/assets/66475393/64bef7b9-566b-4fbc-8a83-f3d393d13873)
![im0 (copy)_depth](https://github.com/BillFSmith/TilingZoeDepth/assets/66475393/8cebe785-a62c-4193-aa0c-7f90b17435ec)

    
