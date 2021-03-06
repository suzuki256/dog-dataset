# Dog dataset for Speak Like a Dog task
This Dog dataset is an example for speak like a dog task.
The speak like a dog task is a human to non-human creature voice conversion task that convert human speech into dog-like speech while preserving the linguistic information and  representing a dog-like elements of the target domain (H2NH-VC). 
H2NH-VC is a task that converts human speech into non-human creature-like speech while preserving linguistic information.  
An ovetview of the speak like a dog task is shown in the following figure.

<p align="center">
  <img src="https://user-images.githubusercontent.com/86309284/160840868-cccbed96-6b2a-46cb-b769-458046b0ed45.png" width="750px">
</p>

The Dog dataset consists of adultdog, puppy, and dogs.
| domain name | detail | the number of data |
| :----: | :----: | :----: |
| adult_dog | Adult dogs voice (low voice) | 792 |
| puppy | Puppies voice (high voice) | 288 |
| dogs | Dogs voice (Consists of the above the domain of adult_dog and puppy) | 1080 |

# Paper
K. Suzuki, S. Sakamoto, T. Taniguchi, and H. Kameoka, “Speak
like a dog: Human to non-human creature voice conversion,”
2022. [(paper)](https://arxiv.org/abs/2206.04780)   

# Converted audio samples
Audio samples are available [here](https://drive.google.com/drive/folders/1aQ5o0Ond50nbAvZsp_me4b97j8VtLYbz?usp=sharing).
  
The directory of audio samples can be viewed as follows:    

```bash
converted _ [ Voice Conversion method ] _ [ audio feature ]  
```
  - [ Voice Conversion method ]  
    * StarGAN = [StarGAN-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/stargan-vc2/index.html)  
    * ACVAE = [Auxiliary classifier Variational Autoencoder-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/acvae-vc3/index.html)  

  - [ audio feature ]  
    * MCC = mel-cepstral coefficients
    * melspec = mel-spectrogram  

The kernel sizes of the discriminator and classifier in StarGAN are as follows.  
```bash
converted _ StarGAN _ [ kernel size ]  
```
  - [ kernel size ]  
    * kernel - 2 = default kernel size - 2
    * kernel - 1 = default kernel size - 1
    * kernel + 1 = default kernel size + 1
    * kernel + 2 = default kernel size + 2  

The audio file reads as follows．  
```bash
[ Source Speaker ] _ to _ [ Target Speaker ]
```
  - For exsample: convert [ adult dog voice ] to [ fkn voice ]
    ```bash
    adultdog_to_fkn  
    ```
# Setup
## Download dataset
  - You can download the dog dataset for Speak Like a dog task [here](https://drive.google.com/drive/folders/1TmG1yjc0_RLUX7U0ZJGLPVWkAwiSkSWY?usp=sharing)  
  - Please collect your own datasets for human speech signals.  
    (We used [ATR digital sound database **set B**](http://www.atr-p.com/products/sdb.html#DIGI) in our paper.)
## preprocess
  Before training your VC model by using the dog dataset, please preprocess the dog dataset to remove fine noise.
  Code that we used in experiment is aveirable [here](https://github.com/suzuki256/dog-dataset/blob/main/extract_features.py). 

# Credit
The following is the original author's credit for the voice data of the dog used．  
The dog dataset used in this experiment consists of the following speech data, from which extremely quiet, loud, and noisy sounds are removed. The dataset is divided into two datasets based on sound height, since there are low sounds like adult dogs and high sounds like puppies.  

The dataset as a whole is available under the terms of the Creative Commons
Attribution-NonCommercial license (https://creativecommons.org/share-your-work/public-domain/cc0/, http://creativecommons.org/licenses/by-nc/3.0/, and https://creativecommons.org/licenses/by/4.0/legalcode).

Please see [License](https://github.com/suzuki256/dog-dataset/blob/main/LICENSE) for an attribution list.

The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or completeness, or absence of errors. 

# Feedback

Please help us improve Dog dataset by sending your feedback to: suzuki.kohei@em.ci.ritsumei.ac.jp
In case of a problem report please include as many details as possible.
