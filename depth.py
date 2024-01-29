#using wsl2 on windows - linux unbutu. replicated the colab environment python 10.3 and all the adds to exact versions as colab


import os
from PIL import Image
import numpy as np    
import cv2
import torch
import torchvision
import timm
from zoedepth.utils.misc import get_image_from_url, colorize   

 

zoe = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
zoe = zoe.to('cuda')
 





# Set the directory containing your images
image_directory = 'your_input_directory'
output_directory = 'your_output_directory'

# Loop through all files in the directory
for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            file_path = os.path.join(image_directory, filename)
            img = Image.open(file_path)
        
    # Process the image here
        print(f'Processing {filename}...')
        print(img.size)  # Check the size of the image

#Generate low resolution image
low_res_depth = zoe.infer_pil(img)

low_res_scaled_depth = 2**16 - (low_res_depth - np.min(low_res_depth)) * 2**16 / (np.max(low_res_depth) - np.min(low_res_depth))

low_res_depth_map_image = Image.fromarray((0.999 * low_res_scaled_depth).astype("uint16"))
low_res_depth_map_image.save('zoe_depth_map_16bit_low.png')

# Generate filters

# store filters in lists
im = np.asarray(img)

tile_sizes = [[4,4], [8,8]]

filters = []

save_filter_images = True

for tile_size in tile_sizes:

        num_x = tile_size[0]
        num_y = tile_size[1]

        M = im.shape[0]//num_x
        N = im.shape[1]//num_y

        filter_dict = {}
        filter_dict['right_filter'] = np.zeros((M, N))
        filter_dict['left_filter'] = np.zeros((M, N))
        filter_dict['top_filter'] = np.zeros((M, N))
        filter_dict['bottom_filter'] = np.zeros((M, N))
        filter_dict['top_right_filter'] = np.zeros((M, N))
        filter_dict['top_left_filter'] = np.zeros((M, N))
        filter_dict['bottom_right_filter'] = np.zeros((M, N))
        filter_dict['bottom_left_filter'] = np.zeros((M, N))
        filter_dict['filter'] = np.zeros((M, N))

        for i in range(M):
          for j in range(N):
              x_value = 0.998*np.cos((abs(M/2-i)/M)*np.pi)**2
              y_value = 0.998*np.cos((abs(N/2-j)/N)*np.pi)**2

              if j > N/2:
                  filter_dict['right_filter'][i,j] = x_value
              else:
                  filter_dict['right_filter'][i,j] = x_value * y_value

              if j < N/2:
                  filter_dict['left_filter'][i,j] = x_value
              else:
                  filter_dict['left_filter'][i,j] = x_value * y_value

              if i < M/2:
                  filter_dict['top_filter'][i,j] = y_value
              else:
                  filter_dict['top_filter'][i,j] = x_value * y_value

              if i > M/2:
                  filter_dict['bottom_filter'][i,j] = y_value
              else:
                  filter_dict['bottom_filter'][i,j] = x_value * y_value

              if j > N/2 and i < M/2:
                  filter_dict['top_right_filter'][i,j] = 0.998
              elif j > N/2:
                  filter_dict['top_right_filter'][i,j] = x_value
              elif i < M/2:
                  filter_dict['top_right_filter'][i,j] = y_value
              else:
                  filter_dict['top_right_filter'][i,j] = x_value * y_value

              if j < N/2 and i < M/2:
                  filter_dict['top_left_filter'][i,j] = 0.998
              elif j < N/2:
                  filter_dict['top_left_filter'][i,j] = x_value
              elif i < M/2:
                  filter_dict['top_left_filter'][i,j] = y_value
              else:
                  filter_dict['top_left_filter'][i,j] = x_value * y_value

              if j > N/2 and i > M/2:
                  filter_dict['bottom_right_filter'][i,j] = 0.998
              elif j > N/2:
                  filter_dict['bottom_right_filter'][i,j] = x_value
              elif i > M/2:
                  filter_dict['bottom_right_filter'][i,j] = y_value
              else:
                  filter_dict['bottom_right_filter'][i,j] = x_value * y_value

              if j < N/2 and i > M/2:
                  filter_dict['bottom_left_filter'][i,j] = 0.998
              elif j < N/2:
                  filter_dict['bottom_left_filter'][i,j] = x_value
              elif i > M/2:
                  filter_dict['bottom_left_filter'][i,j] = y_value
              else:
                  filter_dict['bottom_left_filter'][i,j] = x_value * y_value

              filter_dict['filter'][i,j] = x_value * y_value

        filters.append(filter_dict)

        if save_filter_images:
            for filter in list(filter_dict.keys()):
                filter_image = Image.fromarray((filter_dict[filter]*2**16).astype("uint16"))
                filter_image.save(f'mask_{filter}_{num_x}_{num_y}.png')


# filters second section
compiled_tiles_list = []

for i in range(len(filters)):

        num_x = tile_sizes[i][0]
        num_y = tile_sizes[i][1]

        M = im.shape[0]//num_x
        N = im.shape[1]//num_y

        compiled_tiles = np.zeros((im.shape[0], im.shape[1]))

        x_coords = list(range(0,im.shape[0],im.shape[0]//num_x))[:num_x]
        y_coords = list(range(0,im.shape[1],im.shape[1]//num_y))[:num_y]

        x_coords_between = list(range((im.shape[0]//num_x)//2, im.shape[0], im.shape[0]//num_x))[:num_x-1]
        y_coords_between = list(range((im.shape[1]//num_y)//2,im.shape[1],im.shape[1]//num_y))[:num_y-1]

        x_coords_all = x_coords + x_coords_between
        y_coords_all = y_coords + y_coords_between

        for x in x_coords_all:
            for y in y_coords_all:

                depth = zoe.infer_pil(Image.fromarray(np.uint8(im[x:x+M,y:y+N])))
                


                scaled_depth = 2**16 - (depth - np.min(depth)) * 2**16 / (np.max(depth) - np.min(depth))

                if y == min(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[i]['top_left_filter']
                elif y == min(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_left_filter']
                elif y == max(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[i]['top_right_filter']
                elif y == max(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_right_filter']
                elif y == min(y_coords_all):
                    selected_filter = filters[i]['left_filter']
                elif y == max(y_coords_all):
                    selected_filter = filters[i]['right_filter']
                elif x == min(x_coords_all):
                    selected_filter = filters[i]['top_filter']
                elif x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_filter']
                else:
                    selected_filter = filters[i]['filter']


                print(f"Shape of compiled_tiles section: {compiled_tiles[x:x+M, y:y+N].shape}")
                print(f"Shape of selected_filter: {selected_filter.shape}")
                print(f"Shape of scaled_depth: {scaled_depth.shape}")
                print(f"x: {x}, y: {y}, M: {M}, N: {N}")
                print(f"x_coords_all: {x_coords_all}")
                print(f"y_coords_all: {y_coords_all}")   

                compiled_tiles[x:x+M, y:y+N] += selected_filter * (np.mean(low_res_scaled_depth[x:x+M, y:y+N]) + np.std(low_res_scaled_depth[x:x+M, y:y+N]) * ((scaled_depth - np.mean(scaled_depth)) /  np.std(scaled_depth)))

                

        compiled_tiles[compiled_tiles < 0] = 0
        compiled_tiles_list.append(compiled_tiles)

        tiled_depth_map = Image.fromarray((2**16 * 0.999 * compiled_tiles / np.max(compiled_tiles)).astype("uint16"))
        tiled_depth_map.save(f'tiled_depth_{i}.png')

# combine depth maps
from scipy.ndimage import gaussian_filter
grey_im = np.mean(im,axis=2)

tiles_blur = gaussian_filter(grey_im, sigma=20)
tiles_difference = tiles_blur - grey_im

tiles_difference = tiles_difference / np.max(tiles_difference)

tiles_difference = gaussian_filter(tiles_difference, sigma=40)

tiles_difference *= 5

tiles_difference = np.clip(tiles_difference, 0, 0.999)

mask_image = Image.fromarray((tiles_difference*2**16).astype("uint16"))
mask_image.save('mask_image.png')

combined_result = (tiles_difference * compiled_tiles_list[1] + (1-tiles_difference) * ((compiled_tiles_list[0] + low_res_scaled_depth)/2))/(2)

combined_image = Image.fromarray((2**16 * 0.999* combined_result / np.max(combined_result)).astype("uint16"))

combined_image.save(os.path.join(output_directory, f'{filename.split(".")[0]}_depth.png'))
 

print("Processing ended")