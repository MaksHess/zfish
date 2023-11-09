import itk
import SimpleITK as sitk
import numpy as np
from functools import wraps
import warnings
import h5py

SPACING = (1.0, 1.0, 1.0)  # itk convention: (x, y, z)
DTYPE_CONVERSION = {np.dtype('uint64'): np.dtype('uint16'),
                    np.dtype('uint32'): np.dtype('uint16'),
                    np.dtype('uint16'): np.dtype('uint16'),
                    np.dtype('uint8'): np.dtype('uint8'),
                    np.dtype('int64'): np.dtype('uint16'),
                    np.dtype('int32'): np.dtype('uint16'),
                    np.dtype('int16'): np.dtype('int16'),
                    np.dtype('float64'): np.dtype('float64'),
                    np.dtype('float32'): np.dtype('float32'),
                    np.dtype('float16'): np.dtype('float16'),
                    np.dtype('bool'): np.dtype('uint8'), }


def to_itk(img, spacing=None, conversion_warning=True, pass_everything=False, **kwargs):
    if spacing is None:
        spacing=kwargs.get('element_size_um')
    if isinstance(img, np.ndarray):
        new_dtype = DTYPE_CONVERSION[img.dtype]
        if conversion_warning:
            warnings.warn('Converting {0} to {1}'.format(img.dtype, new_dtype))
        img = img.astype(new_dtype)
        trans_img = itk.GetImageFromArray(img)
        if spacing is None: spacing = SPACING
        trans_img.SetSpacing(spacing)
    elif isinstance(img, sitk.Image):
        trans_img = itk.GetImageFromArray(sitk.GetArrayFromImage(img))
        trans_img.SetOrigin(tuple(img.GetOrigin()))
        if spacing is None: spacing = tuple(img.GetSpacing())
        trans_img.SetSpacing(spacing)
        trans_img.SetDirection(
            itk.GetMatrixFromArray(np.array(img.GetDirection()).reshape((img.GetDimension(), img.GetDimension()))))
    elif isinstance(img, itk.Image):
        trans_img = img
        if spacing is None: spacing = tuple(img.GetSpacing())
        trans_img.SetSpacing(spacing)
    elif isinstance(img, itk.LabelMap.x3):
        filt = itk.LabelMapToLabelImageFilter.LM3IUS3.New(
            img
        )  # BUG: itk does not automatically use US3 if label map has more elements than 255 --> hardcoded .LM3IUS3.
        filt.Update()
        trans_img = filt.GetOutput()
    elif isinstance(img, h5py.Dataset):
        img_dset = img
        img = to_numpy(img_dset)
        new_dtype = DTYPE_CONVERSION[img.dtype]
        if conversion_warning:
            warnings.warn('Converting {0} to {1}'.format(img.dtype, new_dtype))

        trans_img = itk.GetImageFromArray(img.astype(new_dtype))
        if spacing is None: spacing = tuple(
            reversed(img_dset.attrs.get('element_size_um', img_dset.ndim * [1.0]).astype(np.float64)))
        trans_img.SetSpacing(spacing)
    else:
        if pass_everything:
            trans_img = img
        else:
            raise ValueError('Unknown image type: {}'.format(type(img)))
    return trans_img


def to_sitk(img, spacing=None, pass_everything=False, **kwargs):
    if spacing is None:
        spacing=kwargs.get('element_size_um')
    if isinstance(img, itk.LabelMap.x3):
        img = to_itk(img, spacing=spacing, pass_everything=pass_everything)
    if isinstance(img, np.ndarray):
        trans_img = sitk.GetImageFromArray(img)
        if spacing is None: spacing = SPACING
        trans_img.SetSpacing(spacing)
    elif isinstance(img, itk.Image):
        trans_img = sitk.GetImageFromArray(itk.GetArrayFromImage(img))
        trans_img.SetOrigin(tuple(img.GetOrigin()))
        if spacing is None: spacing = tuple(img.GetSpacing())
        trans_img.SetSpacing(spacing)
        trans_img.SetDirection(itk.GetArrayFromMatrix(img.GetDirection()).flatten())
    elif isinstance(img, sitk.Image):
        trans_img = img
        if spacing is None: spacing = tuple(img.GetSpacing())
        trans_img.SetSpacing(spacing)
    elif isinstance(img, h5py.Dataset):
        trans_img = sitk.GetImageFromArray(to_numpy(img))
        if spacing is None: spacing = tuple(
            reversed(img.attrs.get('element_size_um', img.ndim * [1.0]).astype(np.float64)))
        trans_img.SetSpacing(spacing)
    else:
        if pass_everything:
            trans_img = img
        else:
            raise ValueError('Unknown image type: {}'.format(type(img)))
    return trans_img


def to_numpy(img, pass_everything=False, return_metadata=False, **kwargs):
    if isinstance(img, itk.LabelMap.x3):
        img = to_itk(img, pass_everything=pass_everything)
    if isinstance(img, (itk.Image, itk.VectorImage)):
        trans_img = itk.GetArrayFromImage(img)
    elif isinstance(img, sitk.Image):
        trans_img = sitk.GetArrayFromImage(img)
    elif isinstance(img, np.ndarray):
        trans_img = img
    elif isinstance(img, h5py.Dataset):
        trans_img = img[...]
    else:
        if pass_everything:
            trans_img = img
        else:
            raise ValueError('Unknown image type: {}'.format(type(img)))
    if return_metadata:
        metadata = extract_metadata(img)
        return trans_img, metadata
    return trans_img


def extract_metadata(img):
    metadata = dict()
    if hasattr(img, 'GetSpacing'):
        metadata['scale'] = tuple(img.GetSpacing())[::-1]
    if hasattr(img, 'GetOrigin'):
        metadata['origin'] = tuple(img.GetOrigin())[::-1]
    if hasattr(img, 'attrs'):
        metadata = {**metadata, **dict(img.attrs)}
    return metadata


def to_labelmap(img, spacing=None, pass_everything=False):
    filt = itk.LabelImageToLabelMapFilter.New(to_itk(img, spacing=spacing, pass_everything=pass_everything))
    filt.Update()
    return filt.GetOutput()


def at_all(func, output_format, input_format):
    if input_format is None:
        input_format = func.__module__.split('.')[-1].split('_')[0]  # Extract the input image type from the module name
    assert input_format in ['np', 'sitk', 'itk']
    conversion_function = {'np': to_numpy,
                           'sitk': to_sitk,
                           'itk': to_itk,
                           np.ndarray: to_numpy,
                           sitk.Image: to_sitk,
                           itk.Image: to_itk}
    to_in = conversion_function[input_format]  # Pick the appropriate conversion function
    to_out = conversion_function[output_format]

    @wraps(func)
    def all_wrapper(*imgs, **kwargs):
        trans_imgs = []
        trans_kwargs = {}
        for img in imgs:
            trans_imgs.append(to_in(img, pass_everything=True))
        for kwarg in kwargs:
            trans_kwargs[kwarg] = to_in(kwargs[kwarg], pass_everything=True)
        output = func(*trans_imgs, **trans_kwargs)
        if isinstance(output, (tuple, list)):
            return [to_out(e, pass_everything=True) for e in output]
        return to_out(output, pass_everything=True)

    return all_wrapper
