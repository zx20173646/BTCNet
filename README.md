# BTC-Net: Efficient Bit-level Tensor Data Compression Network for Hyperspectral Image
## Introduction
A new bit-level hyperspectral tensor data compression method that combines 
a data-driven quantized neural encoder and channel-wise attention-based enhancement super-resolution.

## Requirements
* Ununtu 18.0 
* python 3.7 
* Pytorch 1.4 

## Training and Testing
Creat ```HSI``` folder and put HSI dataset in, and add corresponding path in the .txt file 
in the ``` testpath ``` and ``` trainpath ```   <br>
Run the ```train.py``` for training and ``` testing.py ``` for testing <br> 

## Semantic Test
In the file ```Classification``` 

```Datasets``` contains the cropped classification datasets
Indian Pines (128×128×172) and Salinas (512×128×172), and 
their corresponding reconstructed data

```checkpointIP``` and ```checkpointSalinas``` contain 10 weight files of the model trained
on the dataset IP and Salinas, respectively.

```logIP``` and ```logSalinas``` contain 5 txt files respectively, recording the results of
10 classification experiments

```IP_ori.txt``` and ```S_ori.txt``` record the accuracies of 10 model weights on the classification dataset
and corresponding values of random seeds (select the training samples randomly)

Make sure you have set the training or testing mode, then
```
python Demo_IP.py
```
or
```
python Demo_S.py
```
to implement the training or testing on the classification datasets