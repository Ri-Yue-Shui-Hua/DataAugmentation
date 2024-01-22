from utils import resize_image, convert_numpy_to_sitk
import SimpleITK as sitk
import os
import numpy as np
from transforms.color_transforms import *
from transforms.spatial_transforms import *
from transforms.crop_and_pad_transforms import *
from transforms.utility_transforms import *
from transforms.sample_normalization_transforms import *
from transforms.noise_transforms import *
from transforms.resample_transforms import *
from transforms.abstract_transforms import Compose
import random


def resize_all_training_data(data_dir, saved_dir, scale=0.5):
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    f_ids = sorted(os.listdir(data_dir))
    count, sum = 0, len(f_ids)
    for f in f_ids:
        count += 1
        fp = os.path.join(data_dir, f)
        print(count,'/',sum, '\t', fp)
        sitk_img = sitk.ReadImage(fp)
        scale_sitk_img = resize_image(sitk_img, scale)
        new_spacing = scale_sitk_img.GetSpacing()
        new_size = scale_sitk_img.GetSize()
        print('New_spacing = ', new_spacing)
        print('New_size = ', new_size)
        saved_fp = os.path.join(saved_dir, f)
        sitk.WriteImage(scale_sitk_img, saved_fp)


def choose_T(T):
    idx = np.random.randint(3)
    t = []
    t.append(Compose(T))
    t.append(OneOfTransform(T))
    t.append(None)
    print(idx)
    return t[idx]


def transform(d, gt):
    T = []
    single_channel_size = d.shape[1:]
    d = d[np.newaxis, :, :, :, :]
    gt = gt[np.newaxis, :, :, :, :]

    # T.append(GaussianNoiseTransform(p_per_sample=0.5))
    T.append(GaussianBlurTransform((0.5, 3), different_sigma_per_channel=False, p_per_sample=0.8))
    # T.append(BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3), per_channel=False, p_per_sample=0.5))
    # T.append(ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.5))
    # T.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1.5), per_channel=False, p_per_channel=0.5,
                                                       # order_downsample=0, order_upsample=3, p_per_sample=0.5,
                                                       # ignore_axes=None))

    # T.append(GammaTransform(gamma_range=(0.7, 1.5), retain_stats=True, p_per_sample=0.5))

    # 0 1 2 for X Y Z (RAS)
    axis = [0, 1, 2]
    T.append(RotateAxisTransform(x_angle_range=(-30, 30), y_angle_range=(-20, 20), z_angle_range=(-30, 30), axes=axis,
                            data_key="data", label_key="gt", p_per_sample=0.9))

    # (0,1) for z, y--> [up_down, front-rear]
    axis = [2]
    T.append(MirrorTransform(data_key='data', label_key='gt', axes=axis))

    direction_and_step_dict = {
        "R": 15,
        "L": 15,
        "A": 15,
        "P": 15,
        "S": 5
        #"I": 5
    }
    T.append(SampleTranslation(data_key='data', label_key='gt', direction_and_step_dict=direction_and_step_dict))

    T = choose_T(T)
    if T is not None:
        out_dict = T(data=d, gt=gt)
        d, gt = out_dict.get('data'), out_dict.get('gt')
    d = d[0]
    gt = gt[0]
    return d, gt


def for_test_load_data(save=False):
    img_path = '/home/algo/data/data/Data_train_scaled_2/0001.nii.gz'
    gt_path = '/home/algo/data/data/Heatmap/Heatmap_2/Train/0001'
    sitk_img = sitk.ReadImage(img_path)
    print('Reading '+img_path+'\tDone!')
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    sitk_img = sitk.GetArrayFromImage(sitk_img)
    sitk_gt = []
    for gt in range(18):
        gt = str(gt+1)+'.nii.gz'
        print('Reading '+ gt +'\tDone!')
        sitk_gt.append(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_path,gt))))

    print('Convert sitkIamge to Numpy array')
    sitk_gt = np.array(sitk_gt)
    print(sitk_gt.max())
    sitk_img = np.array(sitk_img)
    sitk_img = sitk_img[np.newaxis, :, :, :]
    print("DO_Trans")
    img, gt = transform(sitk_img, sitk_gt)
    img = img[0]
    if not save:
        return img, gt
    def write_sitk_img(img, name):
        sitk_img = sitk.GetImageFromArray(img)
        sitk_img.SetSpacing(spacing)
        sitk_img.SetOrigin(origin)
        sitk_img.SetDirection(direction)
        sitk.WriteImage(sitk_img,name)
    print("Save transed_img")
    write_sitk_img(img, 'img_T.nii.gz')
    i = 0
    for g in gt:
        i += 1
        print("Save "+ str(i))
        write_sitk_img(g, 'gt_T_'+str(i)+'.nii.gz')
    return img, gt


def for_test_filp(img_path, saved_name_path, save=False):

    sitk_img = sitk.ReadImage(img_path)
    print('Reading '+img_path+'\tDone!')
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    sitk_img = sitk.GetArrayFromImage(sitk_img)
    print('Convert sitkIamge to Numpy array')

    sitk_img = np.array(sitk_img)
    sitk_img = sitk_img[np.newaxis, np.newaxis, :, :, :]
    print("DO_Trans")
    T = MirrorTransform(data_key='data', label_key='gt', axes=[2], p_per_sample=1)
    out_dict = T(data=sitk_img, gt=None)
    img = out_dict.get('data')[0,0]
    print(img.shape)
    if not save:
        return img

    def write_sitk_img(img, name):
        sitk_img = sitk.GetImageFromArray(img)
        sitk_img.SetSpacing(spacing)
        sitk_img.SetOrigin(origin)
        sitk_img.SetDirection(direction)
        sitk.WriteImage(sitk_img,name)
    print("Save transed_img")
    write_sitk_img(img, saved_name_path)
    return img


def generate_enhanced_data_offline(input_path, output_path, suffix, enhance_type, save_flag):
    print(output_path)
    print(enhance_type)
    file_list = get_files(input_path, suffix)
    for file in file_list:
        sitk_img = sitk.ReadImage(file)
        spacing = sitk_img.GetSpacing()
        origin = sitk_img.GetOrigin()
        direction = sitk_img.GetDirection()
        sitk_img = sitk.GetArrayFromImage(sitk_img)
        sitk_img = np.array(sitk_img)
        sitk_img = sitk_img[np.newaxis, np.newaxis, :, :, :]
        if enhance_type == 0:  # GaussianBlur
            pass
        if enhance_type == 1:  # rotate
            # 为支持旋转，先将数据按照最大尺寸外扩，原数据居中
            # 暂设尺寸 256*278*278
            new_img = -1024 * np.ones((1, 1, 256, 278, 278))
            new_img[:, :, :, 90:186, 90:186] = sitk_img
            sitk_img = new_img
            axis = [0]
            T = RotateAxisTransformExtent(x_angle_range=(0, 45), y_angle_range=(-45, 45), z_angle_range=(-45, 45),
                                         axes=axis, data_key="data", label_key="gt", p_per_sample=1)
            out_dict = T(data=sitk_img, gt=None)
            img = out_dict.get('data')[0, 0]
            print(img.shape)
            pass
        if enhance_type == 2:  # mirror
            T = MirrorTransform(data_key='data', label_key='gt', axes=[2], p_per_sample=1)
            out_dict = T(data=sitk_img, gt=None)
            img = out_dict.get('data')[0, 0]
            print(img.shape)
        if enhance_type == 3:  # translation
            pass
        if not save_flag:
            return img
        print("Save transformed_img")
        saved_name_path = file.replace(input_path, output_path)
        write_sitk_img(img, spacing, origin, direction, saved_name_path)


def get_files(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]


def write_sitk_img(img_arr, spacing, origin, direction, name):
    sitk_img = sitk.GetImageFromArray(img_arr)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)
    sitk.WriteImage(sitk_img, name)


if __name__ == '__main__':
    # img, gt = for_test_load_data(save=True)
    from glob import glob
    # img_path = '/home/algo/data_d/data/Patient_data/test_data/knee_femur/left/BJ_02008_knee.nrrd'
    # img_path = glob("/home/algo/data_d/data/Patient_data/test_data/knee_femur/left/*.nrrd")
    # for f in img_path:
    #     saved_dir = os.path.dirname(f)
    #     saved_name = os.path.basename(f).split('.')[0]+'_flip.nrrd'
    #     saved_name_path = os.path.join(saved_dir, saved_name)
    #     for_test_filp(f, saved_name_path, save=True)
    input_path = r"D:\Dataset\FemurSegmentation\train"
    output_path = r"D:\Dataset\FemurSegmentation\rotate"
    generate_enhanced_data_offline(input_path, output_path, '.nii.gz', 1, 1)



