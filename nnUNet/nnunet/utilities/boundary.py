# -*- coding:utf-8 -*-
from skimage.measure import marching_cubes
import scipy.ndimage as ndimage
from skimage.morphology import cube, erosion, square, dilation
import numpy as np


def get_boundary(mask_array, width=3):
    boundary = np.zeros(mask_array.shape)
    classes = np.unique(mask_array)
    if len(classes) == 1:
        return boundary
    classes.sort()

    for i in range(1, len(classes)):
        level = classes[i-1]
        value = float(classes[i])

        seg_npy = (mask_array > level) * 1.0
        if len(seg_npy.shape) == 2:
            erosion_matrix = square(width)
        elif len(seg_npy.shape) == 3:
            erosion_matrix = cube(width)
        seg_erosion = erosion(seg_npy, erosion_matrix)
        boundary_npy = seg_npy - seg_erosion

        # seg_dilation = dilation(seg_npy, erosion_matrix)
        # boundary_npy = seg_dilation - seg_npy

        boundary[boundary_npy > 0] = value

    return boundary


def get_3d_boundary(mask_array, width=3):
    boundary = np.zeros(mask_array.shape)
    classes = np.unique(mask_array)
    if len(classes) == 1:
        return boundary
    classes.sort()

    for i in range(1, len(classes)):
        level = classes[i-1]
        value = float(classes[i])
        try:
            verts, _, _, _ = marching_cubes(mask_array, level)
        except:
            verts = list()
        for cord in verts:
            cord = cord.astype(np.int32)
            boundary[cord[0], cord[1], cord[2]] = value

    boundary = dilation(boundary, cube(width))
    return boundary


def get_all_boundary(data, width=3):
    data_shape = data.shape
    boundary = np.zeros(data_shape)
    for patch in range(data_shape[0]):
        for modality in range(data_shape[1]):
            boundary[patch, modality] = get_boundary(data[patch, modality], width)
    return boundary


def gen_boundary_from_seg(rseg, width=3):
    rseg_npy = rseg.detach().cpu().numpy()
    seg = rseg_npy.argmax(1)
    boundary = np.zeros([seg.shape[0], 1, *seg.shape[1:]])
    for patch in range(seg.shape[0]):
        boundary[patch, 0] = get_boundary(seg[patch], width)
    return boundary


def gen_all_boundary_from_seg(output, width=3):
    boundary_scales = list()
    for scale in output:
        boundary_scales.append(gen_boundary_from_seg(scale, width))
    return boundary_scales


def generate_boundary_from_segmentation(seg_npy):
    seg_npy = (seg_npy > 0) * 1.0
    seg_erosion = erosion(seg_npy, cube(5))
    boundary_npy = seg_npy-seg_erosion

    return boundary_npy


def generate_segmentation_from_boundary(boundary_npy):
    segmentation = ndimage.binary_fill_holes(boundary_npy)
    return segmentation * 1.0


if __name__ == '__main__':
    # import SimpleITK as sitk
    # img_file = r"/home/sshu/Downloads/case_00000.nii.gz"
    # img_sitk = sitk.ReadImage(img_file)
    # img_npy = sitk.GetArrayFromImage(img_sitk)
    #
    # boundary_npy = generate_boundary_from_segmentation(img_npy)
    # boundary_sitk = sitk.GetImageFromArray(boundary_npy)
    # boundary_sitk.CopyInformation(img_sitk)
    # sitk.WriteImage(boundary_sitk, '/home/sshu/Downloads/case_00000_new_new.nii.gz')

    # seg_npy = generate_segmentation_from_boundary(boundary_npy)
    # seg_sitk = sitk.GetImageFromArray(seg_npy)
    # seg_sitk.CopyInformation(img_sitk)
    # sitk.WriteImage(seg_sitk, '/home/sshu/Downloads/case_00000_seg_recover.nii.gz')

    img_file = r"/home/sshu/Downloads/segmentation.nii.gz"
    import SimpleITK as sitk
    img_sitk = sitk.ReadImage(img_file)
    img_npy = sitk.GetArrayFromImage(img_sitk)
    img_bdr = get_boundary(img_npy)
    bdr_sitk = sitk.GetImageFromArray(img_bdr)
    bdr_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(bdr_sitk, r"/home/sshu/Downloads/4_bdr.nii.gz")

