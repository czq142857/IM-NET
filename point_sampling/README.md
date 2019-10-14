# point sampling code for IM-NET

The code samples 3 levels of resolutions for coarse-to-fine training: 16<sup>3</sup>, 32<sup>3</sup>, 64<sup>3</sup>. But the actual sampled points are from 256<sup>3</sup> voxels.

For 16<sup>3</sup> resolution, we sample 16×16×16 = 4096 points. We first divide the ground truth 256<sup>3</sup> voxels into 16<sup>3</sup> cubes each containing 16<sup>3</sup> voxels. For each cube, if it contains at least one "inside" voxel, we randomly sample one of the "inside" voxels and use the center of that voxel as the sample point. Otherwise we randomly sample one "outside" voxel.

For 32<sup>3</sup> and 64<sup>3</sup> resolution, we sample 4096 and 16384 points respectively. We sample those voxels adjacent to the surface of the shape. Please refer to the code for details.

## Usage

Download the original voxel models from [HSP](https://github.com/chaene/hsp).
Download the rendered views from [3D-R2N2](https://github.com/chrischoy/3D-R2N2).

step 1: change the data directories in *1_check_hsp_mat.py* and run it. Because a few voxel files are corrupted, you need to manually remove them. Their names will be printed out by the script. Also, remove *4a32519f44dc84aabafe26e2eb69ebf4* from category *rifle (04090263)*.

step 2: change the category name and the data directories in *2_gather_256vox_16_32_64.py* and run it for each category. Use *2_test_hdf5.py* to check if the output hdf5 is correct.

step 3: change the category name and the data directories in *3_gather_img.py* and run it for each category.

step 4: run *4_gather_all_vox_img_test.py*, *4_gather_all_vox_img_train.py* to get a hdf5 file containing all categories, or run *4_gather_vox_img_test.py*, *4_gather_vox_img_train.py* to get hdf5 files containing individual categories.


If you have prepared your own voxel data, please modify the code accordingly. The only difference is the code segment for reading voxels.