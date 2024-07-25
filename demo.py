import os
import argparse
import torch
import glob
import numpy as np
from torchvision.ops import masks_to_boxes
from torchvision.transforms import ToPILImage


import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')

parser.add_argument('--img_extn', default="png", help='RGB Image format')


parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--data_dir', default="/home/leo/usman_ws/codes/Fast-SCNN-pytorch/test-data", help='Data directory')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')
parser.add_argument('--cropped', action='store_true', default=False)
parser.add_argument('--masked', action='store_true', default=True)
parser.add_argument('--cpu', dest='cpu', action='store_true', default=True)
parser.set_defaults(cpu=False)


args = parser.parse_args()

def save_image(tensor_image, file_name):
    """
    Save a PyTorch tensor as an image file.

    Args:
    tensor_image (torch.Tensor): The image tensor to save.
    file_name (str): The name of the file to save the image to.
    # Example usage:
    # save_image(batch[0], 'output_image.png')
    """
    # Define the inverse normalization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    # Move the tensor to CPU if it's on GPU
    tensor_image = tensor_image.cpu()

    # Apply the inverse normalization
    tensor_image = inv_normalize(tensor_image)

    # Clip the image to be in the range [0, 1]
    tensor_image = torch.clamp(tensor_image, 0, 1)

    # Convert from (C, H, W) to (H, W, C) and to numpy array
    image = tensor_image.permute(1, 2, 0).numpy()

    # Convert to a PIL image
    image_pil = Image.fromarray((image * 255).astype('uint8'))  # Multiply by 255 if the image is in [0, 1] range

    # Save the image
    image_pil.save(file_name)


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''

     ### Road 0 + Sidewalk 1
    img[img == 1] = 1
    img[img == 0] = 1

    ### building 2 + wall 3 + fence 4
    img[img == 4] = 4
    img[img == 3] = 4
    img[img == 2] = 4

    ### Pole 5 + Traffic Light 6 + Traffic Signal 7
    img[img == 7] = 7
    img[img == 6] = 7
    img[img == 5] = 7
    
    ### vegetation 8 + Terrain 9
    img[img == 9] = 9
    img[img == 8] = 9

    ### Sky 10
    img[img == 10] = 10
    
    ## Person
    img[img == 11] = 11
 
    # cars 13 + truck 14 + bus 15 + train 16
    img[img == 16] = 16
    img[img == 15] = 16
    img[img == 14] = 16
    img[img == 13] = 16
    
    ## Rider 12 + motorcycle 17 + bicycle 18
    img[img == 18] = 18
    img[img == 17] = 18
    img[img == 12] = 18


    ### Don't need, make these 255
    ## Background
    img[img == 19] = 255

    return img

def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    # transform1 = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
    #                                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
    # ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_list = glob.glob(args.data_dir + os.sep + '*.' + args.img_extn)
    print(image_list)

    img = []
    for fname in image_list:
        image = Image.open(fname).convert('RGB')
        image = transform(image).to(device)
        img.append(image)

    batch = torch.stack(img)
    ## Total number of images in batch
    N = batch.size(0)
    
    
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)
    print('Finished loading model!')
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    ### for stage 1 and stage 2, move to the cpu first
    pred_all_c = torch.argmax(outputs[0], 1).cpu().data.numpy()
    ### for stage 3
    pred_all_g = torch.argmax(outputs[0], 1)

    # # Stage 1
    # ## All Predictions of 20 classes
    # Road 0
    # Sidewalk 1
    # building 2
    # wall 3
    # fence 4
    # pole 5
    # traffic light 6
    # traffic signal7
    # vegetation 8
    # terrain 9
    # sky 10
    # person 11
    # rider 12
    # cars 13
    # truck 14
    # bus 15
    # train 16
    # motorcycle 17
    # bicycle 18
    # unknown 19

    ## Uncomment to save the S1 Images
    for jj in range(N):
        ## current image image_list[jj]
        mask = get_color_pallete(pred_all_c[jj], args.dataset)
        outname = os.path.splitext(os.path.split(image_list[jj])[-1])[0] + '-s1.png'
        mask.save(os.path.join(args.outdir, outname))

    ## Stage 2
    # Without merge
    # pred = pred_all_c.astype(np.uint8)
    ## Merge Labels
    pred = relabel(pred_all_c.astype(np.uint8))
    pred_g_merge = relabel(pred_all_g)

    # Uncomment to save the S2 Images
    for jj in range(N):
        ## current image image_list[jj]
        mask = get_color_pallete(pred[jj], args.dataset)
        outname = os.path.splitext(os.path.split(image_list[jj])[-1])[0] + '-s2.png'
        mask.save(os.path.join(args.outdir, outname))

    to_pil = ToPILImage()
    NB = 5
    ## Stage 3
    ## Crop Certain Images / Masks
    ## Multiply with the mask and remove the zero paddings
    for jj in range(N):
        # run for each image
        all_label_mask = pred_g_merge[jj]
        labels_all, label_count_all = all_label_mask.unique(return_counts=True)
        
        # Create a boolean mask where label_count values are >= 5000
        mask_t = label_count_all >= 5000

        # Use the mask to filter labels and label_count
        labels = labels_all[mask_t]
        label_count = label_count_all[mask_t]
        
        masks = all_label_mask == labels[:, None, None]
        regions = masks_to_boxes(masks.to(torch.float32))
        boxesd = regions.to(torch.long)

        image = batch[jj]
        _, H, W = image.shape
        rsizet = transforms.Resize((H, W))
        sub_nodes = []
        
        # run for each label
        for i, label in enumerate(labels):
            binary_mask = (all_label_mask == label).float()
            if args.cropped:
                x_min, y_min, x_max, y_max = boxesd[i]
                masked_image = image[:, y_min:y_max, x_min:x_max]
                if args.masked:
                    # whole->masked->cropped->resize
                    masked_image = masked_image * binary_mask[y_min:y_max, x_min:x_max]
                masked_image = rsizet(masked_image)
            else:
                # masked->whole
                masked_image = image * binary_mask
            
            # sub_nodes.append(masked_image.unsqueeze(0))
            
            # outname = f"{os.path.splitext(os.path.basename(image_list[jj]))[0]}-s3-label{label}.png"
            # save_image(masked_image, os.path.join(args.outdir, outname))

        # Handle the last label separately
        if args.masked:
            binary_mask_all = 1 - (all_label_mask == labels[-1]).float()
            masked_image = image * binary_mask_all
        else:
            masked_image = image
            
        outname = os.path.splitext(os.path.split(image_list[jj])[-1])[0] + f'-s3-label{labels[-1]}.png'
        # save_image(masked_image, os.path.join(args.outdir, outname))

        if len(sub_nodes) < NB:
            bb_x = [
                [0, 0, int(2*W / 3), H],
                [int(W / 3), 0, W, H],
                [0, 0, W, int(2*H / 3)],
                [0, int(H / 3), W, H],
                [int(W / 4), int(H / 4), int(3 * W / 4), int(3 * H / 4)],
            ]
            
            for i in range(len(bb_x) - len(sub_nodes)):
                x_cropped = masked_image[:, bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
                sub_nodes.append(rsizet(x_cropped.unsqueeze(0)))
                outname = f"{os.path.splitext(os.path.basename(image_list[jj]))[0]}-s3-cropped{i}.png"
                save_image(x_cropped, os.path.join(args.outdir, outname))
        
        aa = torch.stack(sub_nodes, 1)
    
    


if __name__ == '__main__':
    demo()
