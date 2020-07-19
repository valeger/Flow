# Flow Model

N-D Flow Model (RealNVP architecture)[]. This model was trained on a low-resolution (32 × 32) version of the CelebA-HQ dataset that has been quantized to 2 bits per color channel (due to lack of more powerful GPU).

## Run

1. Install all dependencies listed in requirements.txt. Note that the model has only been tested in the versions shown in the text file.
2. Set following options:
    * `name` stands for name of the saved model (pt file)
    * `lr` stands for learning rate (default 5e-4) 
    * `epochs` stands for total number of training epochs (100 is pretty good)
    * `gpu` stands for enabling CUDA (True/False)

```bash
cd src && python3 main.py --epochs 100 --gpu True
```
As far as the output, several plots will be saved in `images` directory (train plot + samples from the final trained model + interpolation images). 
Inetrpolation image consists of 5 rows of interpolations between real images in the test set (right and left in a row). 4 intermediate are interpolation ones.
