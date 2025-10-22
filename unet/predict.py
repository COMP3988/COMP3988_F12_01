import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import SimpleITK as sitk

from utils.data_loading import BasicDataset
from model import UNet


def _array2pil_uint8(slice2d: np.ndarray) -> Image:
    """Min-max normalize a 2D array to uint8 and convert to PIL Image."""
    s = slice2d.astype(np.float32)
    s -= s.min()
    denom = (s.max() if s.max() != 0 else 1.0)
    s = (s / denom) * 255.0
    return Image.fromarray(s.astype(np.uint8))


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
            result = mask[0].long().squeeze().numpy()
        else:
            # regression: keep raw float prediction
            result = output[0].squeeze().numpy()

    return result


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        base, ext = os.path.splitext(fn)
        if ext.lower() in {'.mha', '.mhd'}:
            return f'{base}_OUT.mha'
        return f'{base}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    # n_classes hardcoded to 1 from original repo for image -> image regression
    # instead of image segmentation
    net = UNet(n_channels=1, n_classes=1, bilinear=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict, strict=True)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        ext = os.path.splitext(filename)[1].lower()

        if ext in {'.mha', '.mhd'}:
            # Read 3D (or 2D) MetaImage using SimpleITK
            vol_itk = sitk.ReadImage(filename)
            vol_np = sitk.GetArrayFromImage(vol_itk)  # shape: (D, H, W) or (H, W)
            if vol_np.ndim == 2:
                vol_np = vol_np[None, ...]  # make it (D=1, H, W)

            preds = []
            for z in range(vol_np.shape[0]):
                pil_slice = _array2pil_uint8(vol_np[z])
                pred_slice = predict_img(net=net,
                                         full_img=pil_slice,
                                         scale_factor=args.scale,
                                         out_threshold=args.mask_threshold,
                                         device=device)
                preds.append(pred_slice.astype(np.float32))
            pred_vol = np.stack(preds, axis=0)  # (D, H, W)

            # Save output
            out_filename = out_files[i]
            if out_filename.lower().endswith(('.mha', '.mhd')):
                out_itk = sitk.GetImageFromArray(pred_vol)
                out_itk.CopyInformation(vol_itk)
                sitk.WriteImage(out_itk, out_filename)
                logging.info(f'Volume saved to {out_filename}')
            else:
                # Default: also save as NPY alongside any PNG fallback
                npy_path = f"{os.path.splitext(out_filename)[0]}.npy"
                np.save(npy_path, pred_vol)
                logging.info(f'Volume saved to {npy_path}')
        else:
            # Fallback to standard 2D image path via PIL
            img = Image.open(filename)

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)
