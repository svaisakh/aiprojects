# StateFarm Distracted Driver Detection

Made in the quest to win [this](https://www.kaggle.com/c/state-farm-distracted-driver-detection) Kaggle Competition

## Notebooks
The notebooks are intended to be run in the following order:
1. [Preprocessing](Notebooks/Preprocessing.ipynb) - Process the dataset in a Keras friendly format. The creation of a good validation set is particularly important in this competition.
2. [Initiations](Notebooks/Initiations.ipynb) - Try some naive models (linear, feedforward etc.). Get a feel for the data and it's performance.
3. [ConvNet](Notebooks/ConvNet.ipynb) - Try a CNN.
4. [VGG](Notebooks/VGG.ipynb) - Fine-tune the VGG16 model.
5. [ExploreModel](Notebooks/ExploreModel.ipynb) - Explore a model and what it has learnt.
6. [SubmissionMaker](Notebooks/SubmissionMaker.ipynb) - Prepare the submission on the Test set for Kaggle.

## [Follow my Trello Board](https://trello.com/c/DJd7yTUp/22-state-farm-distracted-driver-detection)