# Theano3DCNN
3D Convolutional Neural Network (3DCNN) for classifying videos

# Steps to get this working:
1) Download videos to be processed: e.g. http://www.nada.kth.se/cvap/actions/
2) Create a directory in the data directory corresponding to each label: e.g. boxing
3) Inside each of these new directories create two directories: training and testing.
4) Place avi files inside corresponding directories 
5) Update prepare_settings.py file to point to the right directories ('labels')
6) Run prepare_settings.py

Now you should be able to run the main file:
python main.py results.txt
