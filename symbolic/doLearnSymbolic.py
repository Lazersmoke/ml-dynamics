import os
# Tell tensorflow not to use the GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Tone down warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("Loading TensorFlow...")
import tensorflow as tf

from tensorflow.keras.layers import Reshape, Dense, Flatten, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras import Model

import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.rc('image', cmap='inferno')

import matplotlib.ticker as mplticker

import matlab.engine

from scipy import ndimage, misc, signal, stats

### Settings ###
EPOCHS = 1000

batchSize = 100

### /Settings ###

def intGenerator():
  print("Starting MATLAB...")
  eng = matlab.engine.start_matlab()

  eng.addpath(eng.genpath(r'C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\Utility'))
  eng.addpath(r'C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\src')
  eng.workspace['domain'] = eng.getDomain()
  eng.workspace['endState'] = eng.load("R40_turbulent_state_k1.mat","s")['s']

  exactDefects = 12
  while True:
    print("Integrating...")
    eng.workspace['endState'] = eng.eval("stepIntegrate(domain,endState)")
    defectData = eng.eval("getDefects(domain,endState)")
    yield defectData

fromMatlabDataset = tf.data.Dataset.from_generator(intGenerator,output_signature = tf.TensorSpec(shape=(16,7), dtype=tf.float32)).prefetch(10).take(80).cache(filename='training_data_Symbolic/matlabAutocache')
print(list(fromMatlabDataset.map(lambda x: tf.math.count_nonzero(x[:,0])).as_numpy_iterator()))
print(list(fromMatlabDataset.map(lambda x: x[0,0]).as_numpy_iterator()))

timeSliceSize = 3
positionKern = [0,1,0]
timeDifferenceKern = [-0.5,0,0.5]
timeSecondDifferenceKern = [1,-2,1]

stateDataset = fromMatlabDataset.batch(timeSliceSize,drop_remainder=True).batch(batchSize).shuffle(10000)
trainStates = stateDataset.shard(2,0)
testStates = stateDataset.shard(2,1)

class SymbolicModel(Model):
  def __init__(self):
    super(SymbolicModel,self).__init__()
    #self.inputLayer = Input(shape=(3,))
    self.coeffs = Dense(1,input_dim=4,use_bias=False)

  def call(self,x):
    print(x.shape)
    timeDeriv = tf.tensordot(x,timeDifferenceKern,axes=[-1,-1])
    timeSecondDeriv = tf.tensordot(x,timeSecondDifferenceKern,axes=[-1,-1])
    xTerm = tf.tensordot(x,positionKern,axes=[-1,-1])
    constantTerm = tf.constant(1.0,shape=[x.shape[0]])

    regularizationTerm = tf.stack([0 * constantTerm,constantTerm, 0* constantTerm, 0 * constantTerm],axis=-1)
    sparsificationTermConst = tf.stack([constantTerm * 0,constantTerm * 0, constantTerm * 0, constantTerm],axis=-1)
    sparsificationTermTime = tf.stack([constantTerm ,constantTerm * 0, constantTerm * 0, constantTerm * 0],axis=-1)
    sparsificationTermX = tf.stack([constantTerm * 0,constantTerm * 0, constantTerm , constantTerm * 0],axis=-1)

    return tf.abs(self.coeffs(sparsificationTermX)) + tf.abs(self.coeffs(sparsificationTermTime)) + tf.abs(self.coeffs(sparsificationTermConst)) + 10 * tf.abs(1 - self.coeffs(regularizationTerm)) + tf.abs(self.coeffs(tf.stack([timeDeriv,timeSecondDeriv,xTerm,constantTerm],axis=-1)))

def dataMatchLoss(x,residuals):
  return tf.math.reduce_mean(residuals,axis = -1)

def trainThis(model,trainDir,lossFn,getFeat,getLabel,doViz,trainSet,testSet):
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.05)

  train_loss = tf.keras.metrics.Mean(name='train_loss')

  test_loss = tf.keras.metrics.Mean(name='test_loss')

  @tf.function
  def train_step(original):
    # GradientTape keeps track of gradients so we can backprop
    symOrig = getFeat(original)
    lbl = getLabel(original)
    with tf.GradientTape() as tape:
      reconstructed = model(symOrig, training=True)
      loss = lossFn(lbl, reconstructed)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

  @tf.function
  def test_step(original):
    # training=False disables Dropout layer
    reconstructed = model(getFeat(original), training=False)
    t_loss = lossFn(getLabel(original), reconstructed)

    test_loss(t_loss)

  checkpoint_path = "training_data_" + trainDir + "/cp-{epoch:04d}.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Save the epoch zero weights to ensure the folder is populated
  if not os.path.isdir(checkpoint_dir):
    model.save_weights(checkpoint_path.format(epoch=0))

  latest = tf.train.latest_checkpoint(checkpoint_dir)
  model.load_weights(latest)

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

    for original in trainSet:
      # Progess bar
      print(".",end="",flush=True)
      train_step(original)

    print()

    vizEpochs = 5
    vized = False
    for test_orig in testSet:
      print("'",end="",flush=True)
      test_step(test_orig)
      # Also vizualize on last epoch
      if not vized and ((epoch % vizEpochs == 0) or (epoch == EPOCHS - 1)):
        print("\b\"",end="",flush=True)
        doViz(model,test_orig,epoch)
        model.save_weights(checkpoint_path.format(epoch=epoch))
        vized = True
    print()
    print(
      f'Epoch {epoch}, '
      f'Loss: {train_loss.result()}, '
      f'Test Loss: {test_loss.result()}'
    )

  # Still save weights even if n=0 graphs
  model.save_weights(checkpoint_path.format(epoch=EPOCHS))

def dontViz(x,y,z):
  print([(0.5 * ((y[0,3] - y[0,2]) + (y[0,2] - y[0,1]))).numpy(),(y[0,1] - 2.0 * y[0,2] + y[0,3]).numpy(), (y[0,2]).numpy(), 1])
  print(y[0])
  print(x(y)[0])
  print(x.layers[0].get_weights()[0])
  return

model = SymbolicModel()
trainThis(model,"Symbolic",dataMatchLoss,lambda x: x,lambda o: o,dontViz,trainStates,testStates)
quit()

