import caffe
import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/datascience/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')



plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    raise "Exception"
    


caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB    

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

import os
directory = '/datascience/kaggle/yelp/test_photos/'
directory_out = '/datascience/kaggle/yelp/test_outputs/'
l = sorted(list(os.listdir(directory)))
print(len(l))


partitioned = ((i, l[i:i+50]) for i in xrange(0, len(l), 50))

import cPickle as pickle

for i_data, files in partitioned:
  print ("working on ", str(i_data))	
  data = map(lambda file_name: transformer.preprocess('data', caffe.io.load_image(directory + file_name)), files)  
  
  while np.shape(data)[0] < 50:
    data.append(data[0])

  data = np.reshape(data, [50, 3, 227, 227])

  

  net.blobs['data'].data[...] = data
  out = net.forward()  
  print("Predicted class is #{}.".format(out['prob'][0].argmax()))


  with open(directory_out + 'fc7_' + str(i_data) + '.pickled', 'wb') as handle:
    pickle.dump(net.blobs['fc7'].data[:len(files)], handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory_out + 'fc8_' + str(i_data) + '.pickled', 'wb') as handle:
    pickle.dump(net.blobs['fc8'].data[:len(files)], handle, protocol=pickle.HIGHEST_PROTOCOL)




