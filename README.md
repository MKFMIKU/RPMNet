# RPMNet
Implement of our paper "Residual Refine based Pseudo Multi-frame Network for Efficient Single Image Super-Resolution"

![](docs/Network.png)

## Usage
### Prepare
For more fair comparison with the state-of-art method using [MATLAB](https://www.mathworks.com/products/matlab.html), we use same script to do 
datasets generate and then export them with HDF5 file. So the model in PyTorch could receive same date as previous method done.
- For training, download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and place the folder into `prepare`.
Then run the `generate_train.m` using MATLAB. A big file in train.h5 will appear after it down.

- For testing, download [Testing](http://vllab.ucmerced.edu/wlai24/LapSRN/images/icon_zip.png) datasets, and change the variable `folder` to
the place where test datasets in. Then run it in Matlab. A folder named `testdatasets` will appear.

- For real time loss monitor, we use [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)

### Training
```
usage: train.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
                [--step STEP] [--cuda] [--start-epoch START_EPOCH]
                [--threads THREADS] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [--pretrained PRETRAINED]
                [--dataset DATASET] [--number NUMBER]

RPMNet

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs
  --cuda                Use cuda?
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --momentum MOMENTUM   Momentum, Default: 0.9
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --dataset DATASET     path to load dataset
  --number NUMBER       path to load dataset
```

### Testing
```
usage: test.py [-h] [--cuda] [--model MODEL] [--imagepath IMAGEPATH]
               [--scale SCALE]

SR test

optional arguments:
  -h, --help            show this help message and exit
  --cuda                use cuda?
  --model MODEL         model path
  --imagepath IMAGEPATH
                        image path
  --scale SCALE
```

### Pretrained
https://www.dropbox.com/s/ax5sbrj3loaboab/RPMNet_100.pth?dl=0