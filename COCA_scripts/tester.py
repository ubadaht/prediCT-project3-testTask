import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt

# Check a failing scan — 530977324f9e (16.5%, Moderate)
scan_id = '530977324f9e'
base    = r'...\COCA_output'

warped  = sitk.ReadImage(f'{base}\\experiment3_output\\registration\\{scan_id}\\warped_atlas_seg.nii.gz')
calcium = sitk.ReadImage(f'{base}\\data_resampled\\{scan_id}\\{scan_id}_seg.nii.gz')

warped_arr  = sitk.GetArrayFromImage(warped).astype(np.uint8)
calcium_arr = sitk.GetArrayFromImage(calcium).astype(np.uint8)

spacing     = list(warped.GetSpacing())
spacing_zyx = [spacing[2], spacing[1], spacing[0]]

dist_map = distance_transform_edt(1 - warped_arr, sampling=spacing_zyx)

ca_distances = dist_map[calcium_arr > 0]
print(f'Scan: {scan_id}  (Moderate, 16.5% pass)')
print(f'Vessel voxels      : {int(warped_arr.sum())}')
print(f'Calcium voxels     : {int(calcium_arr.sum())}')
print(f'Warped seg shape   : {warped_arr.shape}')
print(f'Calcium seg shape  : {calcium_arr.shape}')
print(f'Ca distances min   : {ca_distances.min():.1f}mm')
print(f'Ca distances mean  : {ca_distances.mean():.1f}mm')
print(f'Ca distances max   : {ca_distances.max():.1f}mm')
print(f'Within 10mm        : {int((ca_distances<=10).sum())} / {len(ca_distances)}')
print(f'Within 15mm        : {int((ca_distances<=15).sum())} / {len(ca_distances)}')
print(f'Within 20mm        : {int((ca_distances<=20).sum())} / {len(ca_distances)}')
#  import SimpleITK as sitk
# import numpy as np

# # Check the affine transform parameters
# tx = sitk.ReadTransform(r'C:\Users\muham\Desktop\gsoc\code\COCA_output\experiment3_output\registration\8621bbe2d102\transform_affine.tfm')
# print('Transform type:', tx.GetName())
# print('Parameters:', [round(p,4) for p in tx.GetParameters()])
# print()

# # Check fixed image size vs warped seg size
# fixed = sitk.ReadImage(r'C:\Users\muham\Desktop\gsoc\code\COCA_output\data_resampled\8621bbe2d102\8621bbe2d102_img.nii.gz')
# warped = sitk.ReadImage(r'C:\Users\muham\Desktop\gsoc\code\COCA_output\experiment3_output\registration\8621bbe2d102\warped_atlas_seg.nii.gz')
# print(f'Fixed image size : {fixed.GetSize()}')
# print(f'Fixed spacing    : {[round(s,3) for s in fixed.GetSpacing()]}')
# print(f'Warped seg size  : {warped.GetSize()}')
# print(f'Warped spacing   : {[round(s,3) for s in warped.GetSpacing()]}')

# # Check how much of the heart the vessel zone covers
# arr = sitk.GetArrayFromImage(warped)
# print(f'Vessel voxels    : {int((arr>0).sum())}')
# print(f'Total voxels     : {arr.size}')
# print(f'Coverage %       : {100*(arr>0).sum()/arr.size:.2f}%')

# import SimpleITK as sitk
# import numpy as np

# # Check warped vessel mask for this scan
# warped = sitk.ReadImage(r'C:\Users\muham\Desktop\gsoc\code\COCA_output\experiment3_output\registration\8621bbe2d102\warped_atlas_seg.nii.gz')
# arr = sitk.GetArrayFromImage(warped)
# print(f'Warped seg unique values: {np.unique(arr)}')
# print(f'Warped seg vessel voxels: {int((arr > 0).sum())}')
# print(f'Warped seg shape: {arr.shape}')
# print(f'Warped seg max: {arr.max():.4f}')
# print(f'Warped seg values > 0.1: {int((arr > 0.1).sum())}')
# print(f'Warped seg values > 0.5: {int((arr > 0.5).sum())}')
# print(f'Warped seg values = 1.0: {int((arr == 1.0).sum())}')

# import pandas as pd
# df = pd.read_csv(r'C:\Users\muham\Desktop\gsoc\code\COCA_output\data_canonical\tables\split_index.csv')

# new_candidates = ['a67e3ad1f932', 'a066ce96b52f', '530977324f9e', '12441bca5022', '009211cc04c7']
# df.loc[df['scan_id'].isin(new_candidates), 'part2_candidate'] = True

# candidates = df[df['part2_candidate'] == True]
# print(f'Total candidates: {len(candidates)}')
# print(candidates['category'].value_counts().to_string())
# print()
# print(candidates[['scan_id','category','voxels']].to_string(index=False))

# df.to_csv(r'C:\Users\muham\Desktop\gsoc\code\COCA_output\data_canonical\tables\split_index.csv', index=False)
# print('\nCSV updated.')
# import pandas as pd
# df = pd.read_csv(r'C:\Users\muham\Desktop\gsoc\code\COCA_output\data_canonical\tables\split_index.csv')

# # current candidates
# current = df[df['part2_candidate'] == True]
# print(f'Current candidates: {len(current)}')
# print(current['category'].value_counts().to_string())

# # available pool
# available = df[
#     (df['split'] == 'test') &
#     (df['voxels'] > 0) &
#     (df['part2_candidate'] == False)
# ]
# print(f'\nAvailable pool: {len(available)}')
# print(available[['scan_id','category','voxels']].sort_values('category').to_string(index=False))