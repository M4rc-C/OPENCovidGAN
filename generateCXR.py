#! /usr/bin/python3

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys, getopt
from pathlib import Path

def is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def main(argv):
   covid_nbr = 0
   non_covid_nbr = 0
   model_to_load = ''
   try:
      opts, args = getopt.getopt(argv,"hc:n:m:",["covid=","noncovid=","model="])
   except getopt.GetoptError:
      print ('generateCXR.py -m <Model to load> -c <Covid CXR numbers> -n <Non-Covid CXR numbers>')
      sys.exit(2)
   if(len(opts) == 0):
       print ('generateCXR.py -m <Model to load> -c <Covid CXR numbers> -n <Non-Covid CXR numbers>')
       sys.exit(2)
   for opt, arg in opts:
       if opt == '-h':
           print ('generateCXR.py -m <Model to load> -c <Covid CXR numbers> -n <Non-Covid CXR numbers>')
           sys.exit()
       elif opt in ("-c", "--covid"):
           if(not is_integer(arg)):
              print ('Numbers of CXR must be an integer')
              sys.exit(2)
           covid_nbr = int(arg)
       elif opt in ("-n", "--noncovid"):
           if(not is_integer(arg)):
              print ('Numbers of CXR must be an integer')
              sys.exit(2)
           non_covid_nbr = int(arg)
       elif opt in ("-m", "--model"):
           model_to_load = arg
   print ('Covid ', covid_nbr)
   print ('Non-Covid ', non_covid_nbr)
   print ('Model ', model_to_load)

   model = load_model(model_to_load)
   Path("CovidCXR").mkdir(parents=True, exist_ok=True)
   Path("NonCovidCXR").mkdir(parents=True, exist_ok=True)

   for i in range(covid_nbr):
       noise = np.random.normal(0, 0.02, (1, 20000))
       sampled_labels = np.array(0)
       X = model.predict([sampled_labels.reshape((-1, 1)),noise], verbose=0)
       X = (X * 127.5 + 127.5).astype(np.uint8)

       image = Image.fromarray(X[0,:,:,0], "L")
       image.save("CovidCXR/" + str(i + 1) + ".jpeg")

   for i in range(non_covid_nbr):
       noise = np.random.normal(0, 0.02, (1, 20000))
       sampled_labels = np.array(0)
       X = model.predict([sampled_labels.reshape((-1, 1)),noise], verbose=0)
       X = (X * 127.5 + 127.5).astype(np.uint8)

       image = Image.fromarray(X[0,:,:,0], "L")
       image.save("NonCovidCXR/" + str(i + 1) + ".jpeg")

if __name__ == "__main__":
   main(sys.argv[1:])
