import os
import argparse
import torch
import glob
import numpy as np
from torchvision.ops import masks_to_boxes


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

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)



args = parser.parse_args()

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
    # for jj in range(N):
    #     ## current image image_list[jj]    
    #     mask = get_color_pallete(pred_all[jj], args.dataset)
    #     outname = os.path.splitext(os.path.split(image_list[jj])[-1])[0] + '-s1.png'
    #     mask.save(os.path.join(args.outdir, outname))
    
    ## Stage 2
    ## Merge Labels
    pred = relabel(pred_all_c.astype(np.uint8))
    pred_g_merge = relabel(pred_all_g)
    
    ## Uncomment to save the S2 Images
    # for jj in range(N):
    #     ## current image image_list[jj]    
    #     mask = get_color_pallete(pred[jj], args.dataset)
    #     outname = os.path.splitext(os.path.split(image_list[jj])[-1])[0] + '-s2.png'
    #     mask.save(os.path.join(args.outdir, outname))
    
    
    
    ## Stage 3
    ## Crop Certain Images / Masks
    ## Multiply with the mask and remove the zero paddings
    for jj in range(N):
            all_label_mask = pred_g_merge[jj]
            # count ids of 20 cat and dont take zero ids
            labels, label_count = all_label_mask.unique(return_counts=True)
            #dont consider backgroud
            labels = labels[:-1]
            label_count = label_count[:-1]
            
            masks = all_label_mask == labels[:, None, None]
            # create box of that mask using boundries
            regions = masks_to_boxes(masks.to(torch.float32))
            ## scale to feature output
            # boxes = boxes / 16
    
    
    # patch_mask = torch.zeros((H, W)).cuda()
    ## Stage 4
    ## Save the final masks
    # Create a binary mask for each label and apply it to the image
    for jj in range(N): #all images
        for i, label in enumerate(labels): #all labels
            # Create a binary mask for the current label
            binary_mask = (all_label_mask == label).float()
            ### copy from original image
            image = batch[0]
            # Apply the binary mask to the 3D image
            masked_image = image * binary_mask

            # # Save the masked 3D image (example using NIfTI format)
            # masked_image_np = masked_image.cpu().numpy()

            outname = os.path.splitext(os.path.split(image_list[jj])[-1])[0] + f'-s3-label{label}.png'  
                                  
            # Convert the masked image to a PIL image for saving
            masked_image_pil = TF.to_pil_image(masked_image)
            
            # Save the masked image
            masked_image_pil.save(os.path.join(args.outdir, outname))
            
            # # Optionally, convert the binary mask to a PIL image and save it
            # binary_mask_pil = TF.to_pil_image(binary_mask)
            # binary_mask_pil.save(f'binary_mask_label_{label}.png')





    for Nx in range(N):
            img_nodes = []
            for idx in range(len(regions)):
                if (idx == rr_boxes[b_idx] and obj_i[b_idx] > 3000 and len(img_nodes) < self.NB - 2
                    ):
                        patch_mask = patch_mask * 0
                        patch_mask[single_label_mask == obj_ids[b_idx]] = 1
                        patch_maskr = rsizet(patch_mask.unsqueeze(0))
                        patch_maskr = patch_maskr.squeeze(0)
                        boxesd = regions.to(torch.long)
                        x_min, y_min, x_max, y_max = boxesd[b_idx]
                        c_img = x[Nx][:, y_min:y_max, x_min:x_max]
                        resultant = rsizet(c_img)
                        img_nodes.append(resultant.unsqueeze(0))
                        break
    
    
    

    


if __name__ == '__main__':
    demo()
