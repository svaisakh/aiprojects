# CycleGAN

Based on the paper:
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) by
[Jun-Yan Zhu](http://people.eecs.berkeley.edu/~junyanz/), [Taesung Park](https://taesung.me/), [Phillip Isola](http://people.eecs.berkeley.edu/~isola/) and [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)

## Samples
### Horse --> Zebra --> Horse

![Horse2Zebra](Outputs/h2z.png)

### Zebra --> Horse --> Zebra

![Zebraw2Horse](Outputs/z2h.png)

### Apple --> Orange --> Apple

![Apple2Orange](Outputs/a2o.png)

### Orange --> Apple --> Orange

![Zebraw2Orange](Outputs/o2a.png)

## Architecture

![Architecture](Outputs/architecture.jpg)
[Source](https://hardikbansal.github.io/CycleGANBlog/)

Clearly, the model is finding sub-optimal solutions.

In case of the Apple2Orange dataset, the model simply transforms the whole image's colors to map orange to red!


## [Follow my Trello Board](https://trello.com/c/V0RKPw4y/24-cyclegan-unpaired-image-to-image-translation-using-cycle-consistent-adversarial-networks)