# James William Fletcher - June 2021
#   - NPOW POC #3
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from random import seed
from hashlib import sha256
from time import time_ns
from sys import exit

# hyperparameters
# seed(time_ns())
seed(8008135)

# CPU ONLY
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# GPU ONLY
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# GPU INFO
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices())

# total_hashes = 60
# training_iterations = 3333
# activator = 'softsign'
# layers = 3
# batches = 96

# total_hashes = 333
# training_iterations = 33333
# activator = 'softsign'
# layers = 3
# batches = 96

total_hashes = 32768
training_iterations = 63333
activator = 'tanh'
# layers = 3
batches = 9996

# total_hashes = 688595
# training_iterations = 1377190 #33333333
# activator = 'tanh'
# layers = 3
# batches = 9999

# generate test dataset strings
ins = []
outs = []
for x in range(total_hashes):
    ins.append(sha256(open("/dev/urandom","rb").read(20)).hexdigest())
    outs.append(sha256(open("/dev/urandom","rb").read(20)).hexdigest())

# for x in range(total_hashes):
#     print(x, ins[x])
# exit()

# convert hexadecimal characters to normalised embeddings
def toEmbed(ic):
    c = ic.upper()
    if(c == '0'):   return -0.8828125
    elif(c == '1'): return -0.765625
    elif(c == '2'): return -0.6484375
    elif(c == '3'): return -0.53125
    elif(c == '4'): return -0.4140625
    elif(c == '5'): return -0.296875
    elif(c == '6'): return -0.1796875
    elif(c == '7'): return -0.0625
    elif(c == '8'): return  0.0546875
    elif(c == '9'): return  0.171875
    elif(c == 'A'): return  0.2890625
    elif(c == 'B'): return  0.40625
    elif(c == 'C'): return  0.5234375
    elif(c == 'D'): return  0.640625
    elif(c == 'E'): return  0.7578125
    elif(c == 'F'): return  0.875

# variable tolerance here could be additional difficulty
def fromEmbed(f):
    if(f > -1 and f < -0.82421875):             return '0'
    elif(f >= -0.82421875 and f < -0.70703125): return '1'
    elif(f >= -0.70703125 and f < -0.58984375): return '2'
    elif(f >= -0.58984375 and f < -0.47265625): return '3'
    elif(f >= -0.47265625 and f < -0.35546875): return '4'
    elif(f >= -0.35546875 and f < -0.23828125): return '5'
    elif(f >= -0.23828125 and f < -0.12109375): return '6'
    elif(f >= -0.12109375 and f < -0.00390625): return '7'
    elif(f >= -0.00390625 and f <  0.11328125): return '8'
    elif(f >=  0.11328125 and f <  0.23046875): return '9'
    elif(f >=  0.23046875 and f <  0.34765625): return 'A'
    elif(f >=  0.34765625 and f <  0.46484375): return 'B'
    elif(f >=  0.46484375 and f <  0.58203125): return 'C'
    elif(f >=  0.58203125 and f <  0.69921875): return 'D'
    elif(f >=  0.69921875 and f <  0.81640625): return 'E'
    elif(f >=  0.81640625 and f <  1):          return 'F'

# convert strings to embeddings
in_td = []
for i, a in enumerate(ins):
    temp = []
    for i, b in enumerate(a):
        temp.append(toEmbed(b))
    in_td.append(temp)

out_td = []
for i, a in enumerate(ins):
    temp = []
    for i, b in enumerate(a):
        temp.append(toEmbed(b))
    out_td.append(temp)

in_td = np.array(in_td)
out_td = np.array(out_td)

in_td = in_td.reshape((-1, 64))
out_td = out_td.reshape((-1, 64))

# print(in_td.shape)
# exit()

# construct neural network
model = Sequential()
model.add(Dense(64, activation=activator, input_dim=64))
# for x in range(layers-2):
#     model.add(Dense(64, activation=activator))
model.add(Dense(64, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# train network
st = time_ns()
#with tf.device("gpu:0"):
model.fit(in_td, out_td, epochs=training_iterations, batch_size=batches)
timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

# save model
# f = open("nnm/model.txt", "w")
# if f:
#     f.write(model.to_json())
# f.close()

# save weights
# model.save_weights("nnm/weights.h5")

# for layer in model.layers:
#     if layer.get_weights() != []:
#         np.savetxt("nnm/" + layer.name + ".csv", layer.get_weights()[0].flatten(), delimiter=",") # weights
#         np.savetxt("nnm/" + layer.name + "_bias.csv", layer.get_weights()[1].flatten(), delimiter=",") # bias

# weight = model.get_weights()[0]
# np.savetxt('nnm/all_weights.csv', weight, fmt='%s', delimiter=',')
# bias = model.get_weights()[1]
# np.savetxt('nnm/all_bias.csv', bias, fmt='%s', delimiter=',')

# show results
print("Testing Model...")
st = time_ns()
success = 0
failure = 0
for x in range(total_hashes):
    in_p = in_td[x].reshape((-1, 64))
    p = model.predict(in_p)

    ps = []
    for i, c in np.ndenumerate(p):
        ps.append(fromEmbed(c)) # would be faster to skip this step and do the comparisons as embeddings with tolerances

    if ps is not None:
        exp = ins[x].upper()
        prd = ''.join(ps)
        if(exp == prd):
            success += 1
        else:
            failure += 1

tt2 = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(tt2), "seconds")
timetaken += tt2
print("")
print("Total Time Taken:", "{:.2f}".format(timetaken), "seconds")
print("total_hashes:", total_hashes)
print("training_iterations:", training_iterations)
print("activator:", activator)
# print("layers:", layers)
print("batches:", batches)
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")
print("Success/Failure =", success, "/", failure)

# in production I would presume most miners to skip the final
# verification stage on the high probability, of their training
# having had sufficient iterations to be correct.
# - hence there is no need to optimise the 'show results' loop

# setting up tensorflow to use a gpu can be a pain, but this
# should help for ubuntu users:

# sudo apt install nvidia-cuda-toolkit
# sudo apt install nvidia-driver-465
# sudo apt install python3
# sudo apt install python3-pip
# sudo pip3 install --upgrade pip
# pip3 install tensorflow-gpu
# sudo ln -s /usr/lib/x86_64-linux-gnu/libcusolver.so.10.6.0.245 /usr/lib/x86_64-linux-gnu/libcusolver.so.11

# Then download and install the runtime and developer debs from here:
# https://developer.nvidia.com/rdp/cudnn-download

# nvidia-driver-465 has to be installed after nvidia-cuda-toolkit or the toolkit overwrites
# the nvidia-smi program which is particularly helpful. nvidia-settings is also noteworthy.

# .. although training on a GPU does not seem to yeild a significant performance gain.

