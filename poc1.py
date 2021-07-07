# James William Fletcher - June 2021
#   - NPOW POC #1
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from random import randint
from sys import exit

in1s = "5426ca398cdf52e5acdcc5b7f07f69ab16ccee91c361e649b4d91399048d87c0"
out1s = "976d37abad4f34c2be446ea09a1b237246c9755c8dc87e461d52a995bf6ff98d"

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

in1 = []
for i, c in enumerate(in1s):
    in1.append(toEmbed(c))

out1 = []
for i, c in enumerate(out1s):
    out1.append(toEmbed(c))

in1 = np.array(in1)
out1 = np.array(out1)

#print(in1)
#exit()

model = Sequential()

model.add(Dense(64, activation='tanh', input_dim=64))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='linear'))

# optim = keras.optimizers.SGD(learning_rate=0.00001, momentum=0, nesterov=False)

model.compile(optimizer='adam', loss='mean_squared_error')

in1 = in1.reshape((-1, 64))
out1 = out1.reshape((-1, 64))

model.fit(in1, out1, epochs=50, batch_size=3)

print("expected:\n", out1)

p = model.predict(in1)
print("prediction:\n", p)

ps = []
for i, c in np.ndenumerate(p):
    ps.append(fromEmbed(c))

exp = out1s.upper()
prd = ''.join(ps)
print("expected: ", exp)
print("predicted:", prd)
if(exp == prd):
    print("SUCCESS")
else:
    print("FAILED")
