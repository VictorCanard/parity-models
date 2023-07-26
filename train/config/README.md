# Training configurations
This directory contains example configuration files used in training in the
parity models framework.

The training framework uses JSON files to configure training runs. These JSON
files are parsed by the [train.py](../train.py) script to launch
training jobs.

## Configuration parameters
We will describe each configuration parameter by walking through an example
configuration file, [mnist.json](mnist.json):
```json
{
  "num_epoch": 500,
  "k_vals": [2, 3, 4],
  "enc_dec_types": [["coders.summation.AdditionEncoder", "coders.summation.SubtractionDecoder"]],
  "datasets": ["mnist"],
  "base_parity_model": ["base-mlp", "resnet18"],
  "losses": ["mse"],
  "train_encoder": false,
  "train_parity_model": true,
  "train_decoder": false
}
```

* **num_epoch**: The number of epochs (passes over the entire training dataset)
                 to be made during training
* **k_vals**: A list of values of the coding parameter k to explore.
              This controls how many inputs are encoded together into a single
              parity input. This repository has been tested with values of k of
              2, 3, and 4.
* **enc_dec_types**: A list of pairs of encoders and decoders to train with.
                     Each pair will be trained. Names of encoders are to be
                     the string path that one would use when importing an
                     encoder or decoder in Python from the [train](..)
                     directory. Encoders and decoders that come with this
                     repository are implemented in [coders](../coders).
* **datasets**: A list of datasets to train on. Current datasets available
                are mnist, fashion-mnist, cifar10, cifar100, cat_v_dog, and
                gcommands.
* **base_parity_models**: A list of pairs of neural network architectures to consider for the base
              & parity models. The models that are currently supported
              are implemented in [base_models](../base_models) and include
              base-mlp, resnet18, resnet152, vgg11, lenet5.
* **losses**: A list of loss values to use. The only supported loss function
              at this time are [torch.nn.MSELoss](https://pytorch.org/docs/master/generated/torch.nn.MSELoss.html),
              [loss.kldiv.KLDivLoss](../loss/kldiv.py), and [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html). The latter two are only supported if `train_encoder` and `train_decoder`
              (described below) are set to `true`.
* **train_encoder**: Whether the encoder should be trained. Currently, this should
                     be set to `true` only if the encoder is [coders.mlp.MLPEncoder](../coders/mlp.py)
                     or [coders.conv.ConvEncoder](../coders/conv.py).
* **train_parity_model**: Whether the parity model should be trained. If this
                          is set to `false`, the parity model will be identical
                          to the base model (i.e., have the same parameters).
* **train_decoder**: Whether the decoder should be trained. Currently, this should
                     be set to "true" only if the encoder is [coders.mlp.MLPDecoder](../coders/mlp.py).

All combinations of configuration parameters in lists will be executed one
after the other. In the MNIST example above, the configuration with `k=2` would
be run first, then with `k=3`, then with `k=4`.
