# Dog dataset for Speak Like a Dog task
The Dog dataset is an example for speak like a dog task.
The speak like a dog task is a human to non-human creature voice conversion task that convert human speech into dog-like speech while preserving the linguistic information and  representing a dog-like elements of the target domain (H2NH-VC). 
H2NH-VC is a task that converts human speech into non-human creature-like speech while preserving linguistic information.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86309284/160269270-ea2ea5ef-66cc-49e4-951f-d296b311790b.png" width="750px">
</p>

The Dog dataset consists of adultdog, puppy, and dogs.
| domain name | detail | the number of data |
----|---- |----|
| adultdog | Adult dogs voice (low voice) | 792 |
| puppy | Puppies voice (high voice) | 288 |
| dogs | Dogs voice (Consists of the above the domain of adultdog and puppy) | 1080 |

# Audio samples
Audio samples are available [here](https://drive.google.com/drive/folders/1pQcEvnD6_r9F12U7iImevYdzoTTfpr7G?usp=sharing).
  
The directory of audio samples can be viewed as follows:    

```bash
converted _ [ Voice Conversion method ] _ [ audio feature ]  
```
  - Voice Conversion method  
    * sgvc=[StarGAN-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/stargan-vc2/index.html)  
    * acvae=[Auxiliary classifier Variational Autoencoder-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/acvae-vc3/index.html)  

  - audio feature  
    * mccs=mel-cepstral coefficients
    * melspec=mel-spectrogram  

The kernel sizes of the discriminator and classifier in StarGAN are as follows.  
```bash
converted _sgvc _ [ kernel size ]  
```
  - kernel size  
    * k-2=default kernel size - 2
    * k-1=default kernel size - 1
    * k1=default kernel size + 1
    * k2=default kernel size + 2  

The audio file reads as follows．  
```bash
[ Source Speaker ] _ to _ [ Target Speaker ]
```
  - For exsample: adultdog to fkn voice conversion 
    ```bash
    adultdog_to_fkn  
    ```

# Download dataset
  You can download dataset for Speak Like a dog task [here](https://drive.google.com/drive/folders/1TmG1yjc0_RLUX7U0ZJGLPVWkAwiSkSWY?usp=sharing)  

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
