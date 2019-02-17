# feature-transfer

about Transfer learning  (survey Pan)

The objective of this work is to investigate different transfer learning setups for the target task of lane following using only frontal images and a discretized control (forward, left, right). The source task is image classification for the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).

#### Techniques for feature transfer learning

+ __frozen__ - the weights learned on the source (base) network are copied to the target network and they're kept frozen during training (i.e. the errors from training the target network are not backpropagated to these layers)
+ __fine-tuning__ - the weights of the target network are initialized with the weights from the base network and are fine-tuned during training

[//]: # "images taken with a frontal camera mounted on top of a toy remote control car"
[//]: # "undersampling method was used to balance the self-driving dataset"

## Experiments

Following [\[1\]](#References) the first thing to do is to train **base** networks for both tasks. For this study the chosen architecture was [MobileNet v1](https://arxiv.org/abs/1704.04861). Since there is a mismatch in the number of classes (1000 for ImageNet and 3 for lane following), only the top layers will be different throughout this experiment.

+ __baseA__ - MobileNet model, with weights pre-trained on ImageNet (1000 categories and 1.2 million images) 
+ __baseB__ - MobileNet model, with weights randomly initialized trained on [self-driving dataset](https://github.com/paulaksm/self_driving_data) (3 categories and 50k images)

BaseA model is not available in this repository because it's downloaded directly from Keras Applications.

BaseB achieved 91% accuracy on validation and was trained for 15 epochs. BaseB model is available at `transfer/trained-models/baseB.h5`.

There is also a baseB+ model, available on `transfer/trained-models/baseBfinetune.h5`,  with weights initialized using the ImageNet pre-trained model and fine-tuned on task B. This model achieved 92% accuracy on validation and was trained for 15 epochs.

#### Self-driving dataset distribution info

##### Training (total: 56172)

+ left = 16141
+ right = 10425
+ forward = 29606

##### Validation (total: 7022)

+ left = 2034
+ right = 1336
+ forward = 3652

##### Test (total: 4086)

+ left = 1362
+ right = 1362
+ forward = 1362

#### Experiment setup

The MobileNet architecture consists of one regular convolutional layer (input), 13 depthwise separable convolutional blocks, one fully-connected layer and a softmax layer for classification.

### Selffer Network (BnB)

The selffer network investigates the interaction between layers; frozen some and training others for the same task. 

Example: `B3B` - the first 3 layers/blocks are copied from `baseB.h5` model and frozen. The following layers are initialized randomly and trained on the self-driving dataset. This network is a control for the `transfer network`.

| model    |   B1B  |   B2B  |   B3B  |   B4B  |   B5B  |   B6B  |   B7B  |   B8B  |   B9B  |  B10B  |  B11B  |  B12B  |  B13B  |
|----------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| val_acc  | 0.9101 | 0.9226 | 0.9209 | 0.9213 | 0.9211 | 0.9213 | 0.9206 | 0.9209 | 0.9105 | 0.9129 | 0.9165 | 0.9166 | 0.9145 |
| test_acc | 0.9138 | 0.9231 | 0.9207 | 0.9199 | 0.9158 | 0.9211 | 0.9248 | 0.9260 | 0.9062 | 0.9106 | 0.9138 | 0.9133 | 0.9150 |

As shown in the table, there is no fragile co-adapted features on successive layers. (pesquisar mais sobre esse tópico)

### Selffer Network (BnB+)



### Transfer Network (AnB)

| model | A1B | A2B | A3B | A4B | A5B | A6B | A7B | A8B | A9B | A10B | A11B | A12B | A13B |
|----------------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| val_acc | 0.7497 | 0.6583 | 0.7359 | 0.5601 | 0.7383 | 0.7221 | 0.7477 | 0.7375 | 0.7618 | 0.7769 | 0.7073 | 0.7445 | 0.5934 |
| test_acc | 0.7126 | 0.6987 | 0.7704 | 0.6830 | 0.7212 | 0.7816 | 0.7080 | 0.7398 | 0.7650 | 0.7709 | 0.6189 | 0.7241 | 0.4676 |
| mean_time/epoch | 471s | 386s | 311s | 288s | 264s | 249s | 250s | 233s | 205s | 188s | 182s | 177s | 149s |
| early_stopping?(ep.) | yes(6) | yes(4) | yes(4) | yes(4) | yes(7) | yes(6) | yes(6) | yes(5) | yes(4) | yes(7) | yes(5) | yes(4) | yes(6) |

### Transfer Network (AnB+)

| model    |  A1B+  |  A2B+  |  A3B+  |  A4B+  |  A5B+  |  A6B+  |  A7B+  |  A8B+  |  A9B+  |  A10B+ |  A11B+ |  A12B+ |  A13B+ |
|----------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| val_acc  | 0.9218 | 0.9289 | 0.9269 | 0.9248 | 0.9109 | 0.9250 | 0.9246 | 0.9307 | 0.9238 | 0.9332 | 0.9279 | 0.9223 | 0.9339 |
| test_acc | 0.9087 | 0.9216 | 0.9216 | 0.9219 | 0.9023 | 0.9270 | 0.9260 | 0.9253 | 0.9243 | 0.9312 | 0.9312 | 0.9194 | 0.9309 |


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

#### Downloading dataset

The script to download the dataset for **task B** is available at [self-driving data repository](https://github.com/paulaksm/self_driving_data). 

You can place the created folder `data/` wherever you like, but _don't rename the `.npy` files_.

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References
[1] [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)

[2] [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526)