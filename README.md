## SRGAN_Cramer
We apply Cramer GAN to SRGAN, which is a GAN based super resolution algorithm.

***This repo was forked from @zsdonghao 's [tensorlayer/srgan](https://github.com/tensorlayer/srgan) repo. We implement the Cramer GAN for SRGAN based on this repo. We changed some loss functions in the original code so as to determine the Cramer distance. Many thanks to @zsdonghao again, for his great effort on the reimplementation.***

### SRGAN Architecture

TensorFlow Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"][1](https://arxiv.org/abs/1609.04802)

### Cramer GAN

The Cramer GAN[2](https://arxiv.org/abs/1705.10743) was designed to minimize the Cramer distance (distance between the real and generated samples). It has the same properties as Wasserstein GAN[3](https://arxiv.org/abs/1701.07875), but also has the unbiased sample gradients. Hence, it provides a more stable learning, and avoids the loss that converge to a wrong minimum. Fig. 1(extract from [2]) shows the general idea of algorithm.

![](https://github.com/hcleung35/SRGAN_Cramer/blob/master/cramer.png)

### Our Contribution

- 1. Apply the Cramer framework to SRGAN
- 2. Eliminate the pre-train process since the Cramer GAN converges to the right minimum.
- 3. Change the output of discriminator(SRGAN_d) to energy loss vector instead of logit.

### Training Dataset and Pre-trained VGG

- 1. Please download a pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).
- 2. Please download the DIV2K high resolution images dataset. in [here] (https://data.vision.ee.ethz.ch/cvl/ntire17//).

### Result


  
### Run

The environment is [TensorFlow](https://www.tensorflow.org) 2.1.0 and the [TensorLayer](https://github.com/tensorlayer/tensorlayer) 2.0.0+.

You can configure your training dataset in config.py 
```python
config.TRAIN.img_path = "your_image_folder/"
```

- Start training.
```bash
python train_cam.py --mode srgan_cam
```

- Start evaluation. 
```bash
python myeval.py --mode myevaluate #(image no.)
```

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [The Cramer Distance as a Solution to Biased Wasserstein Gradients](https://arxiv.org/abs/1705.10743)
* [3] [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

### Author
- [H.C. Leung](https://github.com/hcleung35)
- [H.T. Wong]()

### License

- For academic and non-commercial use only.
- For commercial use, please contact tensorlayer@gmail.com.