## SRGAN Cramer with video super-resolution
We apply Cramer GAN to SRGAN, which is a GAN based super-resolution algorithm.
Based on our trained model, we provide an application that upscales a low-resolution video to a high-resolution one.

Please watch the video in 1080p and theater mode
[![](https://github.com/hcleung35/SRGAN_Cramer/blob/master/imgs/yt.png)](https://www.youtube.com/watch?v=SOtYEjqSciQ&t=19s)

### SRGAN Architecture

TensorFlow Implementation of "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" [1]
@zsdonghao 's [tensorlayer/srgan](https://github.com/tensorlayer/srgan) provides a perfect reimplementation for [1].
This repo is developed based on his repo while we implement the Cramer GAN for SRGAN. We changed some loss functions in the original code so as to determine the Cramer distance. Many thanks to @zsdonghao again, for his great effort on the reimplementation.

### Cramer GAN

The Cramer GAN [2] was designed to minimize the Cramer distance (distance between the real and generated samples). It has the same properties as Wasserstein GAN [3], but also has the unbiased sample gradients. Hence, it provides a more stable learning, and avoids the loss that converge to a wrong minimum. Fig. 1 (extract from [2]) shows the general idea of algorithm.

![](https://github.com/hcleung35/SRGAN_Cramer/blob/master/imgs/cramer.png)

### Our Contribution

- Apply the Cramer framework to SRGAN
- Eliminate the pre-train process since the Cramer GAN converges to the right minimum.
- Change the output of discriminator(SRGAN_d) to energy loss vector instead of logit.
- Apply our model to video super-resolution.

### Training Dataset and Pre-trained VGG

- Please download a pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).
- Please download the DIV2K high resolution images dataset in [here](https://data.vision.ee.ethz.ch/cvl/ntire17).

### Result

The upper batch are the low resolution images. The lower batch are the super resolution images.

![](https://github.com/hcleung35/SRGAN_Cramer/blob/master/imgs/result1.png)

![](https://github.com/hcleung35/SRGAN_Cramer/blob/master/imgs/result2.png)

![](https://github.com/hcleung35/SRGAN_Cramer/blob/master/imgs/result3.png)

### Run

The environment is [TensorFlow](https://www.tensorflow.org) 2.1.0 , [TensorLayer](https://github.com/tensorlayer/tensorlayer) 2.0.0+
, [Opencv](https://pypi.org/project/opencv-python/) 4.2.0+ and [moviepy](https://pypi.org/project/moviepy/) 1.0.3.


You can configure your training dataset in config.py.
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

- Start video super-resolution
```bash
python sr_video.py (video_name).mp4
```

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [The Cramer Distance as a Solution to Biased Wasserstein Gradients](https://arxiv.org/abs/1705.10743)
* [3] [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

### Author
- [H.C. Leung](https://github.com/hcleung35)
- [H.T. Wong](https://github.com/htwong-ai)

### License

- For academic and non-commercial use only.