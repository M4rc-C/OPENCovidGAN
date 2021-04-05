### About

It is simply a project that I realized in the field of my studies. It is a free implementation of the CovidGAN project that I made from the  research report that is available here: https://arxiv.org/abs/2103.05094

CovidGAN is an ACGAN designed to generate a chest x-ray dataset to allow better detection of Covid-19 via CNNs.



### Overview of the OPENCovidGAN training process

<img src="https://media.giphy.com/media/LFOlOmZENc1aJjogtu/giphy-downsized-large.gif" style="zoom:80%;" />

### Install dependencies

Run :

`pip3 install -r requirements.txt`



### Launch training

For launch training run:

`python3 OPENCovidGAN.py`



### Generate CXR

For generate CXR run:

`python3 generateCXR.py -m <Model to load> -c <Covid CXR numbers> -n <Non-Covid CRX numbers>`

This will create two folders "CovidCXR" and "NonCovidCXR" containing the desired number of generated CXR.



### Ressources

###### Used dataset

https://drive.google.com/file/d/1PhcfFPCpFHx4nGLqp8R3elpi8zEo1O49/view?usp=sharing

###### Trained model

https://drive.google.com/file/d/1OJkTLGujFIiOzenG4WUuFbWOrdOf6oOq/view?usp=sharing