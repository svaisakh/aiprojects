## How to use

### Training

1. Place the desired style image as style.jpg in this directory.
2. Place the training content images in a folder images/training.
3. Run train.py. This will train the model for one epoch.
4. If you desire more epochs (I've not found the need for more than 3 epochs for any style), run it again.

### Sampling

1. Place the weights as weights.h5 in this directory. Note that if you've run training.py following the above steps, this step should not be necessary.
2. Place the desired content images in a folder images/raw/samples.
3. Run painter.py
4. The stylized images should be in images/paintings with the same file names.