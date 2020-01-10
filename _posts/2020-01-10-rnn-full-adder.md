---
layout: post
title:  "Binary Full Adder using Custom RNN in Keras"
date:   2020-01-09 21:00:00 +0530
categories: ml
---


<a href="https://colab.research.google.com/github/luckykadam/adder/blob/master/rnn_full_adder.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab
</a>
<a href="https://github.com/luckykadam/adder/blob/master/rnn_full_adder.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub
</a>

## Introduction
It's rare to encounter a situation where LSTM/GRU might not be the choice of RNN cell. How hard it would be to identify and work-around the problem? Read on to find out.

In this notebook, we will emulate Binary Full Adder using RNN in Keras. We will see:

1. why at some situations LSTM/GRU might not be the most optimal choice
2. how to write custom RNN layer

## Background

In the <a href="https://luckykadam.github.io/ml/2019/12/17/full-adder.html">previous post</a> we developed a small neural network to simulate binary full adder. We analysed all the parameters learnt, plotted decision hypersurfaces and drew the circuit. Later, we observed how much the usage pattern resembled Recurrent Neural Network. So, lets see how to achieve the same objective using RNN.

## Full Adder

A Full Adder can perform an addition operation on three bits. The full adder produces a sum of three inputs and carry value. The carry value can then be used as input to the next full adder.

Using this unit in repeatition, two binary numbers of arbitrary length can be added.

<img height="220" src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Full-adder_logic_diagram.svg/800px-Full-adder_logic_diagram.svg.png">


## RNN eumlation

The structure of Full Adder is very similar to <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">how RNN works</a>. We can expolit this similarity.

In the current context, we want RNN cell with output and state, both of dimension 1, but representing **independent** information.

Let's have a look at common choices of RNN cells:

### GRU (Gated Recurrent Unit)
<img src="{{site.baseurl}}/assets/rnn_full_adder/gru.png" height="240">
<br>
This cell has output and state of same size, but they are not independent. Infact, output and state are the same vector in GRU. Hence, not 
suitable here.

### LSTM (Long-Short Term Memory)
<img src="{{site.baseurl}}/assets/rnn_full_adder/lstm.png" height="240">
<br>
This cell produces two states (cell state and hidden state) of different sizes, and an output. Hidden state and output are the exact same vector, which means only cell state is useful in the next iteration (being independent of output). If we configure the network to have cell state and output as size 1, and train, it might learn to ignore the redundant hidden state. But, there will still be parameter corresponding to hidden state, learning thing which are eventually ignored. Hence, we might actually achieve the objective, but with useless parameters learnt, which doesn't look optimal.

So, I guess we will have to define out own custom RNN cell. Lets jump right in ;)

## Implementation

We are going to use Keras (`tf.keras` from Tensorflow 2.0) and it's `keras.layers.RNN` API to implement out RNN.


```python
# only for Google Colab compatibiity
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
```


```python
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers, activations


print(tf.__version__)
# set random seed to get reproducible results
np.random.seed(0)
tf.random.set_seed(1)
```

    2.0.0


## Dataset creation

Dataset can be easily prepared by randomly generating two sets of numbers and adding these sets across to get expected result. We generate numbers with a limit to number of bits in binary representation: `max_bit`.


```python
max_bits = 8
n_samples = 100000
```


```python
# samples in decimal form
samples = np.random.randint(np.power(2, max_bits-1), size=(n_samples, 2))
summed_samples = np.sum(samples, axis=1)
```


```python
# convert samples to binary representation
samples_binary_repr = [[np.binary_repr(a, width=max_bits), np.binary_repr(b, width=max_bits)] for a,b in samples]
summed_binary_repr = [np.binary_repr(c, width=max_bits) for c in summed_samples]

```


```python
x_str = np.array([[list(a), list(b)] for a, b in samples_binary_repr])
y_str = np.array([list(c) for c in summed_binary_repr])
```


```python
# flip binary representation to get increasing significant bit
x_flipped = np.flip(x_str, axis=-1)
y_flipped = np.flip(y_str, axis=-1)
```


```python
# convert string to numbers
x = np.transpose((x_flipped == '1')*1, axes=(0, 2, 1))
y = (y_flipped == '1')*1
```


```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
```

## RNN Cell

Each cell will be a neural network we came up with in the <a href="https://luckykadam.github.io/ml/2019/12/17/full-adder.html">previous post</a>.

The RNN cell will have: three inputs (i<sup>th</sup> bit of the 2 numbers and previous carry), one hidden layer (3 neurons) and one output layer (2 neurons). Out of two output bits, we want one to be a part of the answer and other to be input (carry) to the next RNN cell.

We extend `keras.layers.Layer` to define the custom RNN cell. To define any custom layer we need to follow these steps:

1. define `__init__()` to initialize some object level constants. Keras requires you to declare `units` variable: dimension of the output.
2. define `build()` to initialize all the trainable parameters and set `built=True`.
3. define `call()` to compute the output (and state) using input and parameters.


```python
class FullAdderCell(layers.Layer):
    def __init__(self, hidden_units, **kwargs):
        super(FullAdderCell, self).__init__(**kwargs)
        self.units = 1
        self.state_size = 1
        self.hidden_units = hidden_units

    def build(self, input_shape):
        self.hidden_kernel = self.add_weight(shape=(input_shape[-1] + self.state_size, self.hidden_units),
                                      initializer='uniform',
                                      name='hidden_kernel')
        self.hidden_bias = self.add_weight(shape=(1, self.hidden_units),
                                      initializer='uniform',
                                      name='hidden_bias')
        self.output_kernel = self.add_weight(shape=(self.hidden_units, self.units + self.state_size),
                                      initializer='uniform',
                                      name='output_kernel')
        self.output_bias = self.add_weight(shape=(1, self.units + self.state_size),
                                      initializer='uniform',
                                      name='output_bias')
        self.built = True

    def call(self, inputs, states):
        x = tf.concat([inputs, states[0]], axis=-1)
        h = tf.keras.activations.tanh(tf.matmul(x, self.hidden_kernel) + self.hidden_bias)
        o_s = tf.keras.activations.sigmoid(tf.matmul(h, self.output_kernel) + self.output_bias)
        output = o_s[:, :self.units]
        state = o_s[:, self.units:]
        return output, [state]

```

## Model

`Sequential` API can be used to define the model. We need to wrap the RNN cell with `keras.layer.RNN`, to get an RNN layer. We set `return_sequences=True`, because we want to collect the bits produced by RNN cell at each step.


```python
model = tf.keras.Sequential(name='full_adder')
model.add(layers.RNN(FullAdderCell(3), return_sequences=True, input_shape=(None, 2)))

model.summary()
```

    Model: "full_adder"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    rnn (RNN)                    (None, None, 1)           20        
    =================================================================
    Total params: 20
    Trainable params: 20
    Non-trainable params: 0
    _________________________________________________________________


## Loss function

At each step, only one bit is produced, giving the output of shape `(batch_size, max_bits, 1)`, hence we use `binary_crossentropy` loss function.


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Training


```python
model.fit(x_train, y_train, batch_size=32, epochs=5)
scores = model.evaluate(x_test, y_test, verbose=2)
```

    Train on 90000 samples
    Epoch 1/5
    90000/90000 [==============================] - 14s 151us/sample - loss: 0.6916 - accuracy: 0.5087
    Epoch 2/5
    90000/90000 [==============================] - 12s 138us/sample - loss: 0.3403 - accuracy: 0.8986
    Epoch 3/5
    90000/90000 [==============================] - 13s 147us/sample - loss: 0.0644 - accuracy: 1.0000
    Epoch 4/5
    90000/90000 [==============================] - 13s 144us/sample - loss: 0.0131 - accuracy: 1.0000
    Epoch 5/5
    90000/90000 [==============================] - 14s 154us/sample - loss: 0.0034 - accuracy: 1.0000
    10000/1 - 1s - loss: 0.0017 - accuracy: 1.0000


## Testing

Let's generate two random numbers in range (0, 2<sup>max_bits-1</sup>), predict their sum using our network, and compare it with actual sum.


```python
max_bits = 8

a = np.random.randint(np.power(2, max_bits-1))
b = np.random.randint(np.power(2, max_bits-1))

a_bin = np.float32(1) * (np.flip(list(np.binary_repr(a, width=max_bits)), axis=-1) == '1')
b_bin = np.float32(1) * (np.flip(list(np.binary_repr(b, width=max_bits)), axis=-1) == '1')

print('a: {}, b: {}'.format(a, b))
print('binary representations -> a: {}, b: {}'.format(a_bin, b_bin))

a_b = np.stack((a_bin, b_bin), axis=-1).reshape(1,-1,2)
print('a_b: {}'.format(a_b))
```

    a: 48, b: 10
    binary representations -> a: [0. 0. 0. 0. 1. 1. 0. 0.], b: [0. 1. 0. 1. 0. 0. 0. 0.]
    a_b: [[[0. 0.]
      [0. 1.]
      [0. 0.]
      [0. 1.]
      [1. 0.]
      [1. 0.]
      [0. 0.]
      [0. 0.]]]



```python
predictions = model(a_b).numpy().flatten()

summed_bin = 1 * (predictions > 0.5)
summed = np.packbits(np.flip(summed_bin , axis=-1))[0]
print('predictions: {}'.format(predictions))
print('binary representations -> summed: {}'.format(summed_bin))
print('summed: {}'.format(summed))
```

    predictions: [0.00120685 0.998754   0.00120774 0.998754   0.9987539  0.9987539
     0.00120774 0.00120688]
    binary representations -> summed: [0 1 0 1 1 1 0 0]
    summed: 58


## Result

Voila! Our network worked perfectly. Its amazing, how easily we created a custom RNN layer using `keras.layer.RNN` and `keras.layer.Layer` APIs.

## Conclusion

Frameworks like Keras, Tensorflow and PyTorch give us power to experiment at such speed and effeciecy. Combine it with python's flexibility, and you have got a game-changer in AI department.

## References:

1. <https://en.wikibooks.org/wiki/Digital_Electronics/Digital_Adder>
2. <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
3. <https://www.tensorflow.org/guide/keras/rnn>
4. <http://dprogrammer.org/rnn-lstm-gru>
