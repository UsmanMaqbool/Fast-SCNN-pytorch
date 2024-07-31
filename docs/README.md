## Training-Fast-SCNN

Change the normalization in train.py file

```py
transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
```

## Datasets
- You can download [cityscapes](https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Note: please download [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).

### Setting up

```sh



# Download using the above links
cd /home/leo/usman_ws/datasets
unzip ~/Downloads/leftImg8bit_trainvaltest.zip -d citys/
unzip ~/Downloads/gtFine_trainvaltest.zip -d citys/

# Dataset should me .datasets/citys/
cd /home/leo/usman_ws/codes/Fast-SCNN-pytorch
mkdir datasets
ln -s /home/leo/usman_ws/datasets/citys/ citys
```


## Training-Fast-SCNN
- By default, we assume you have downloaded the cityscapes dataset in the `./datasets/citys` dir.
- To train Fast-SCNN using the train script the parameters listed in `train.py` as a flag or manually change them.
```Shell
python train.py --model fast_scnn --dataset citys 
```