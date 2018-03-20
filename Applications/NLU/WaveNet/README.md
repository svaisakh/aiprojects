# Wavenet

_Note: This is a failed implementation_

Based on the paper:
[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) by
 [Aaron van den Oord](https://avdnoord.github.io/homepage/), [Sander Dieleman](http://benanne.github.io/), [Heiga Zen](https://uk.linkedin.com/in/heiga-zen-b1a64b3), [Karen Simonyan](https://scholar.google.co.uk/citations?user=L7lMQkQAAAAJ), [Oriol Vinyals](https://research.google.com/pubs/OriolVinyals.html), [Alex Graves](https://en.wikipedia.org/wiki/Alex_Graves_(computer_scientist)), [Nal Kalchbrenner](https://www.nal.ai/), [Andrew Senior](https://research.google.com/pubs/author37792.html) and [Koray Kavukcuoglu](http://koray.kavukcuoglu.org/)
 
![Dilated Convolutions](Outputs/dilated.gif)
 
Source: [Blog post](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
 
## Architecture
 
![Architecture](Outputs/architecture.png)
 
 Source: Paper
 
## Notebooks
The notebooks are intended to be run in the following order:
1. [Preprocess](Notebooks/Preprocess.ipynb) - Download and extract the LJ Speech Dataset.
2. [Train](Notebooks/Train.ipynb) - Create and train the model.
3. [Sample](Notebooks/Sample.ipynb) - Sample audio from a trained model.

## [Follow my Trello Board](https://trello.com/c/Hm6cyS1i/4-wavenet-a-generative-model-for-raw-audio)