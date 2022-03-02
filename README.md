# 犬声合成をした音声サンプルは以下より再生可能 (A sample of the synthesized dog voice can be played below ）．  
ディレクトリの見方は次のようになっている．  
converted\_ [ Voice Conversion method ] \_ [ audio feature ]  
* sgvc=Cross-Entropy-StarGAN
* acvae=Auxiliary classifier Variational Autoencoder-VC
* mccs=mel-cepstral coefficients
* melspec=mel-spectrogram  
Cross-Entropy-StarGANのdiscriminatorとclassifierのカーネルサイズに関しては，
converted\_sgvc\_ [ k〇 ]  
となっており，〇の中が
* k_2=k-2
* k_1=k-1
* k1=k+1
* k2=k+2
を示している．  
https://drive.google.com/drive/folders/1pQcEvnD6_r9F12U7iImevYdzoTTfpr7G?usp=sharing
# 以下に使用した犬の音声データの原作者クレジットを載せる．
※本実験で使用した犬のデータセットは，下記の音声データの内，音が極端に小さい音声，大きい音声，ノイズが大きい音声を取り除いたもので構成されており，成犬のような低い音声と子犬のような高い音声があるため，音の高さでデータセットを分けている．  
(1) C. Jacoby and J. P. Bello, ”A Dataset and Taxonomy for Urban Sound Research”,
22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.  
(2) The pet dog sound events data sets are available online at https://github.com/kyb2629/pdse.  
(3) Naoya Takahashi, Michael Gygli, Beat Pfister and Luc Van Gool, ”Deep Convolutional
Neural Networks and Data Augmentation for Acoustic Event Recognition”,
Proc. Interspeech 2016, San Fransisco  
(4) K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of
the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015. [DOI:
http://dx.doi.org/10.1145/2733373.2806390]  
(5) Suh, Dan. (2019). BarkMeowDB - WAV Files of Dogs and Cats (0.2) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.3563990  
(6) Juan Merie Venter, Dog Bark.wav, https://freesound.org/s/327666/  
(7) Astounded, Christopher J Astbury, Switzerland  
(8) apolloaiello, https://freesound.org/s/276267/  
(9) ”Dog Barking, Single, A.wav” by InspectorJ (www.jshaw.co.uk) of Freesound.org  
(10) ”LBS FX DOG Small Alert Bark001.wav” by LittleBigSounds, https://freesound.org/s/163459/  
(11) ”Blossom Bark 29sec mix.wav” by Zajjman, https://freesound.org/s/456943/  
(12) ”Dog / Dog Bark Staffordshire Bullterrier” by Anton, https://freesound.org/s/157322/  
(13) ”Animals - Dogs / Dog bark 2” by jorickhoofd, https://freesound.org/s/160093/  
(14) ”dog barking / single dog bark 1” by crazymonke9  
(15) ”Dog sounds / DogGrunt.wav” by TobiasKosmos, https://freesound.org/s/163282/  
(16) ”Dogs playing / Dog bark with echo and splash” by Vortichez, https://freesound.org/s/545440/  
(17) ”Jazz the Dog (Howl & Bark) / Jazz the Dog Howl Bark (128).wav” by delphidebrain,
https://freesound.org/s/236051/  
(18) ”DOGS everywhere sounds / 06923 watching bad dog.wav” by Robinhood76, https://freesound.org/s/380350/  
(19) ”Dog Sounds / ANGRY DOG BARK SNARL Flat.wav” by deleted user 3424813,https://freesound.org/s/260776/  
(20) ”dog Max / dog max whine howl bark.wav” by lewisinheaven, https://freesound.org/s/389581/  
(21) ”Dogs / 6 week beagle bark.wav” by Tito Lahaye, https://freesound.org/s/86279/  
(22) ”Sled Dogs in Colorado / Mostly One DogWoofing.wav” by be-steele, https://freesound.org/s/369647/  
(23) ”BARKING DOG.wav” by tsakanemashaba, https://freesound.org/s/501678/  
(24) ”Dog barks” by iamkaylagreen, https://freesound.org/s/460253/  
(25) ”Gurgorecordings / Dog stimulation” by vikuserro, https://freesound.org/s/341041/  
(26) ”Dog sound” by nathymunoz, https://freesound.org/s/586213/  
(27) ”Dog barking” by BiancaBothaPure, https://freesound.org/s/365669/  
(28) ”dog barking” by Eelke, https://freesound.org/s/593653/  
(29) ”House/Unprocessed / dog barking.wav” by manda g, https://freesound.org/s/55005/  
(30) ”My dog talking” by blimp66, https://freesound.org/s/554260/  
(31) ”dogs.wav” by Gianna dc, https://freesound.org/s/595452/  
(32) ”Barking Dog.wav” by Dstruct, https://freesound.org/s/87779/  
(33) ”Dog panting” by stevenpam, https://freesound.org/s/215765/  
(34) ”Dog Barks”, by exuberate, https://freesound.org/s/254191/  
(35) ”Dog Bark”, by wxuberate, https://freesound.org/s/254192/  
(36) ”Crits / Dog scream”, by olzzy, https://freesound.org/s/220629/  
(37) ”Dog Barking SFX.wav”, by CameronPheiffer, https://freesound.org/s/443302/  
(38) ”big dog barking.wav” by buzzmsc, https://freesound.org/s/428860/  
(39) ”Sound Effects / Small Dog Barking.wav”, by bennathanras, https://freesound.org/s/607436/  
(40) ”Dog barking OWI.wav” by Livwagner847, https://freesound.org/s/591786/  
(41) ”Singl dog barking” by nomerodin1, https://freesound.org/s/474158/  
(42) ”Singl dog barking” by nomerodin1, https://freesound.org/s/474159/  
(43) ”Ambience / Distant Dog Barks” by andersmmg, https://freesound.org/s/518976/  
(44) ”DOGS BARKING SOUND BULGARIA.wav” by savataivanov, https://freesound.org/s/380827/  
(45) ”BigDogBarking 02.wav” by www.bonson.ca, https://freesound.org/s/24965/  
(46) ”PlayfulDogBarks.WAV” by bmccoy2, https://freesound.org/s/128891/https://freesound.org/s/128891/  
(47) ”dog barking.wav” by paraesius, https://freesound.org/s/107190/  
(48) ”Dog Barking.wav” by CashCarlo, https://freesound.org/s/594354/  
(49) ”dog barking inside a room” by saphe, https://freesound.org/s/187378/  
(50) ”animals dog bark springer spaniel 001.wav” by soundscalpel.com, https://freesound.org/s/110389/  
(51) ”The Big Circle / barking dog” by adejabor, https://freesound.org/s/157950/  
(52) ”Dog Barking” by avakas, https://freesound.org/s/171464/  
(53) ”Large dog whimpering whining crying.wav” by Astounded, https://freesound.org/s/556035/  
(54) ”dogs.wav” by pyoorgoodshed, https://freesound.org/s/426646/  
(55) ”dog barking.wav” by VincentKurtAnderes, https://freesound.org/s/492211/  
(56) ”Barking Dog.wav” by Blu 150058, https://freesound.org/s/326207/  
(57) ”Animal noises (made by myself) / dog barking.wav” by Jerry520, https://freesound.org/s/47926/  
(58) ”Dog Playfully Growling / ginger1.wav” by tomc1985, https://freesound.org/s/84649/  
(59) ”A Single Barking Dog Clip.mp3” by EdTK, https://freesound.org/s/513990/  
(60) ”dog.bark.01.flac” by dobroide, https://freesound.org/s/7913/  
(61) ”Small dog barking” by alec havinmaa, https://freesound.org/s/444310/  
(62) ”Animals, beasts &amp; monsters / dog yowl.wav” by xpoki, https://freesound.org/s/432753/  
(63) ”Small dog crying .wav” by Pablobd, https://freesound.org/s/511011/  
(64) ”Dog Barking 2.wav” by Benboncan, https://freesound.org/s/105088/  
(65) ”Aggressive Guard Dogs” by Oneirophile, https://freesound.org/s/142344/  
(66) ”Doggy Style / Robinhood76 00889 watching dog 2 dog2pup.wav” by Timbre, https://freesound.org/(67) ”Dog Barking Short Sound Effect” by SOANAC, https://www.freesoundslibrary.com/dogbarking-short-sound-effect/#google vignette  
(68) ”Angry Dog Barking Sound Effect” by SPANAC, https://www.freesoundslibrary.com/angrydog-barking-sound-effect/　
(69) ”Nervous Dog Barking Sound Effect” by SPANAC, https://www.freesoundslibrary.com/nervousdog-barking-sound-effect/  
(70) ”Angry Dog Barking Close Sound Effect” by SPANAC, https://www.freesoundslibrary.com/angrydog-barking-close-sound-effect/  
(71) ”Aggressive Dog Barking Sound Effect” by SPANAC, https://www.freesoundslibrary.com/aggressivedog-barking-sound-effect/  
(72) ”Dog Barking Sound Effect” by SPANAC, https://www.freesoundslibrary.com/dogbarking-sound-effect/  
(73) ”Barking Dog Sound”, by SPANAC, https://www.freesoundslibrary.com/barkingdog-sound/  
(74) ”Large Dog Bark Once Sound Effect” by SPANAC, https://www.freesoundslibrary.com/largedog-bark-once-sound-effect/  
(75) ”Dog Growling Sound Effect” by SPANAC, https://www.freesoundslibrary.com/doggrowling-sound-effect/
