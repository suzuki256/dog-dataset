# Dataset for Speak Like a Dog task


Audio samples are available [here](https://drive.google.com/drive/folders/1pQcEvnD6_r9F12U7iImevYdzoTTfpr7G?usp=sharing).
  
The view of the directory is as follows.    

```bash
converted _ [ Voice Conversion method ] _ [ audio feature ]  
```
Voice Conversion method  
* sgvc=StarGAN-VC
* acvae=Auxiliary classifier Variational Autoencoder-VC  

audio feature  
* mccs=mel-cepstral coefficients
* melspec=mel-spectrogram  

The kernel sizes of the discriminator and classifier in StarGAN are as follows.  
```bash
converted _sgvc _ [ kernel size ]  
```
kernel size  
* k-2=default kernel size - 2
* k-1=default kernel size - 1
* k1=default kernel size + 1
* k2=default kernel size + 2  

The audio file reads as follows．  
```bash
[ Source Speaker ] _ to _ [ Target Speaker ]
```
For exsample:  
```bash
adultdog_to_fkn  
```
Voice Conversion from FKN into adult dog  

# Set up dog dataset
  You can download dataset for Speak Like a dog task:  
  [here](https://drive.google.com/drive/folders/1TmG1yjc0_RLUX7U0ZJGLPVWkAwiSkSWY?usp=sharing)  
  
# Credit
The following is the original author's credit for the voice data of the dog used．  
The dog dataset used in this experiment consists of the following speech data, from which extremely quiet, loud, and noisy sounds are removed. The dataset is divided into two datasets based on sound height, since there are low sounds like adult dogs and high sounds like puppies.  

The dataset as a whole is available under the terms of the Creative Commons
Attribution-NonCommercial license (https://creativecommons.org/share-your-work/public-domain/cc0/, http://creativecommons.org/licenses/by-nc/3.0/, and https://creativecommons.org/licenses/by/4.0/legalcode).

Please see [License](https://github.com/suzuki256/dog-dataset/blob/main/LICENSE) for an attribution list.

The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or completeness, or absence of errors. 

# Feedback

Please help us improve UrbanSound8K by sending your feedback to: justin.salamon@nyu.edu or justin.salamon@gmail.com
In case of a problem report please include as many details as possible.
