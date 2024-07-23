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

parser.add_argument('--img_extn', default="jpg", help='RGB Image format')


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
parser.add_argument('--cropped', action='store_true', default=True)
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

    ### Road + Sidewalk
    img[img == 1] = 1
    img[img == 0] = 1

    ### building + Wall + fence
    img[img == 4] = 4
    img[img == 3] = 4
    img[img == 2] = 4

    ### Pole + Traffic Light + Traffic Signal
    img[img == 7] = 7
    img[img == 6] = 7
    img[img == 5] = 7
    
    ### Terrain + vegetation 
    img[img == 9] = 9
    img[img == 8] = 9

    ### Sky
    img[img == 10] = 10
 
    ### Don't need
    img[img == 19] = 255
    img[img == 18] = 255
    img[img == 17] = 255
    img[img == 16] = 255
    img[img == 15] = 255
    img[img == 14] = 255
    img[img == 13] = 255
    img[img == 12] = 255
    img[img == 11] = 255
   
    return img

def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
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

    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)
    print('Finished loading model!')
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    ### for stage 1 and stage 2, move to the cpu first
    pred_all_c = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    ### for stage 3
    pred_all_g = torch.argmax(outputs[0], 1).squeeze(0)

    ## Total number of images in batch
    N = batch.size(0)

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
        all_label_mask = pred_g_merge[jj]
        # count ids of 20 cat and dont take zero ids
        labels, label_count = all_label_mask.unique(return_counts=True)
        
        # dont consider backgroud
        # labels = labels[:-1]
        # label_count = label_count[:-1]

        masks = all_label_mask == labels[:, None, None]
        # create box of that mask using boundries
        regions = masks_to_boxes(masks.to(torch.float32))
        ## scale to feature output
        # boxes = boxes / 16
        ### Crop patches
        boxesd = regions.to(torch.long)

        ## Stage 4
        ## Save the final masks
        # Create a binary mask for each label and apply it to the image
        ### copy from original image
        image = batch[jj]
        _, H, W = image.shape
        rsizet = transforms.Resize((H, W))
        sub_nodes = []
        for i, label in enumerate(labels): #all labels
            # Create a binary mask for the current label
            binary_mask = (all_label_mask == label).float()
            if i < len(labels)-1:
                # Apply the binary mask to the 3D image
                if args.cropped:
                    # Crop the image to the bounding box of the binary mask
                    x_min, y_min, x_max, y_max = boxesd[i]
                    masked_image = image[:, y_min:y_max, x_min:x_max] * binary_mask[y_min:y_max, x_min:x_max]
                    masked_image = rsizet(masked_image)              
                else:
                    masked_image = image * binary_mask
                sub_nodes.append(masked_image.unsqueeze(0))
            else:
                # create mask with all label present except 
                binary_mask_all = 1 - binary_mask
                masked_image = image * binary_mask_all
            outname = os.path.splitext(os.path.split(image_list[jj])[-1])[0] + f'-s3-label{label}.png'  
            ### Save all patches
            save_image(masked_image, os.path.join(args.outdir, outname))

        if len(sub_nodes) < NB:
            bb_x = [
                [int(W / 4), int(H / 4), int(3 * W / 4), int(3 * H / 4)],
                [0, 0, int(W / 3), H],
                [0, 0, W, int(H / 3)],
                [int(2 * W / 3), 0, W, H],
                [0, int(2 * H / 3), W, H],
            ]
            
            for i in range(len(bb_x) - len(img_nodes)):
                x_cropped = masked_image[:, bb_x[i][1] : bb_x[i][3], bb_x[i][0] : bb_x[i][2]]
                sub_nodes.append(rsizet(x_cropped.unsqueeze(0)))
        aa = torch.stack(sub_nodes, 1)


if __name__ == '__main__':
    demo()
