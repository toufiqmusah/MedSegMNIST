# Progress

## Why brain3d_224.npz (665 MB) > brain3d_native.npz (432 MB)

Native volumes (240×240×155, no resampling) have 23% **more** voxels than 224×224×144, yet compress to **38% less** space. Reason: TorchIO's resample → resize chain introduces interpolation artifacts / high-frequency noise that deflate cannot compress as well. Native data is the original smooth MRI acquisition and compresses much more efficiently.
