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

batchSize = 32

exactDefects = 12

### /Settings ###

def intGenerator():
  print("Starting MATLAB...")
  eng = matlab.engine.start_matlab()

  eng.addpath(eng.genpath(r'C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\Utility'))
  eng.addpath(r'C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\src')
  eng.workspace['domain'] = eng.getDomain()
  eng.workspace['endState'] = eng.load("R40_turbulent_state_k1.mat","s")['s']

  while True:
    defectDatums = []
    while True:
      print("Integrating...")
      defectData = eng.eval("getDefects(domain,endState)")
      if len(defectData) == exactDefects:
        defectDatums.append(defectData)
        eng.workspace['endState'] = eng.eval("stepIntegrate(0.05,domain,endState)")
      else:
        print("Skipping region with (" + str(len(defectData)) + ") defects...")
        eng.workspace['endState'] = eng.eval("stepIntegrate(2,domain,endState)")
        break
    if len(defectDatums) > 0:
      yield defectDatums

timeSliceSize = 3
fromMatlabDataset = (tf.data.Dataset
  .from_generator(intGenerator,output_signature = tf.TensorSpec(shape=(None,exactDefects,7), dtype=tf.float32))
  .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x).batch(timeSliceSize,drop_remainder=True))
  .prefetch(10)
  .take(500)
  .cache(filename='training_data_DefectData/matlabAutocache'))
#print(fromMatlabDataset)
#print(list(fromMatlabDataset.map(lambda x: tf.math.count_nonzero(x[:,:,0])).as_numpy_iterator()))
#print(list(fromMatlabDataset.map(lambda x: x[:,0,0]).as_numpy_iterator()))
print(fromMatlabDataset)

positionKern = [0,1,0.0]
timeDifferenceKern = [-0.5,0,0.5]
timeSecondDifferenceKern = [1,-2,1.0]

def buildTerm(x,exponents):
  timeKernels = tf.stack([positionKern,timeDifferenceKern],axis=0)
  # x is [batchSize,timeSliceSize,exactDefects,vars]
  # timeKernels is [orders,timeSliceSize]
  timeDerivs = tf.tensordot(x,timeKernels,axes=[1,1])
  # timeDerivs is [batchSize,exactDefects,vars,orders]
  # exponents is [vars,orders]
  exponents = tf.expand_dims(tf.expand_dims(exponents,axis=0),axis=0)
  term = tf.math.reduce_prod(tf.math.pow(timeDerivs,exponents),axis=[-1,-2])
  # term is [batchSize,exactDefects]
  return term

def termMatrix(varCount,opCount,v,o):
  m = tf.one_hot(varCount * v + o,varCount * opCount,dtype=tf.float32)
  m = tf.reshape(m,[varCount,opCount])
  return m

stateDataset = fromMatlabDataset.batch(batchSize).shuffle(10000)
trainStates = stateDataset.shard(2,0)
testStates = stateDataset.shard(2,1)

class SymbolicModel(Model):
  def __init__(self):
    super(SymbolicModel,self).__init__()
    #self.inputLayer = Input(shape=(3,))
    self.varCount = 7
    self.coeffs = Dense(1,input_dim=4,use_bias=False)

  def call(self,x):
    lossAmt = 0

    locx = buildTerm(x,termMatrix(self.varCount,2,0,0))
    diffsX = tf.tile(tf.expand_dims(locx,axis=-1),[1,1,locx.shape[1]])
    diffsX = diffsX - tf.transpose(diffsX,perm=[0,2,1])

    locy = buildTerm(x,termMatrix(self.varCount,2,1,0))
    diffsY = tf.tile(tf.expand_dims(locy,axis=-1),[1,1,locy.shape[1]])
    diffsY = diffsY - tf.transpose(diffsY,perm=[0,2,1])

    dxdt = buildTerm(x,termMatrix(self.varCount,2,0,1)) # dx/dt
    dydt = buildTerm(x,termMatrix(self.varCount,2,1,1)) # dy/dt
    #timeDeriv = tf.tensordot(x,timeDifferenceKern,axes=[-3,-1])
    #timeSecondDeriv = tf.tensordot(x,timeSecondDifferenceKern,axes=[-3,-1])
    #xTerm = tf.tensordot(x,positionKern,axes=[-3,-1])
    #constantTerm = tf.constant(1.0,shape=xTerm.shape)

    #regularizationTerm = tf.stack([0 * constantTerm,constantTerm, 0* constantTerm, 0 * constantTerm],axis=-1)
    #sparsificationTermConst = tf.stack([constantTerm * 0,constantTerm * 0, constantTerm * 0, constantTerm],axis=-1)
    #sparsificationTermTime = tf.stack([constantTerm ,constantTerm * 0, constantTerm * 0, constantTerm * 0],axis=-1)
    #sparsificationTermX = tf.stack([constantTerm * 0,constantTerm * 0, constantTerm , constantTerm * 0],axis=-1)
    matchedDxdt = tf.broadcast_to(tf.expand_dims(dxdt,-1),diffsX.shape)
    matchedDydt = tf.broadcast_to(tf.expand_dims(dydt,-1),diffsY.shape)
    composite = tf.stack([matchedDxdt,matchedDydt,diffsX,diffsY],-1)

    normTerm = tf.expand_dims(tf.convert_to_tensor([1.0,0,0,0]),0)
    normalizationLoss = tf.abs(1.0 - self.coeffs(normTerm))

    sparse1 = tf.expand_dims(tf.convert_to_tensor([0,0,0,1.0]),0)
    sparse2 = tf.expand_dims(tf.convert_to_tensor([0,0,1.0,0]),0)
    sparse3 = tf.expand_dims(tf.convert_to_tensor([0,1.0,0,0]),0)
    sparsificationLoss = tf.abs(self.coeffs(sparse1)) + tf.abs(self.coeffs(sparse2)) + tf.abs(self.coeffs(sparse3))

    lossAmt = normalizationLoss + sparsificationLoss + 100 * tf.math.reduce_mean(tf.abs(self.coeffs(composite)),axis = [-1,-2])

    return lossAmt

    #return tf.abs(self.coeffs(sparsificationTermX)) + tf.abs(self.coeffs(sparsificationTermTime)) + tf.abs(self.coeffs(sparsificationTermConst)) + 10 * tf.abs(1 - self.coeffs(regularizationTerm)) + lossAmt

def dataMatchLoss(x,residuals):
  return tf.math.reduce_mean(residuals * residuals,axis = [-1,-2])

def trainThis(model,trainDir,lossFn,getFeat,getLabel,doViz,trainSet,testSet):
  optimizer = tf.keras.optimizers.Adam()

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
        #model.save_weights(checkpoint_path.format(epoch=epoch))
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
  #print(y[0])
  #print(x(y)[0])
  print(x.layers[0].get_weights()[0])
  return

model = SymbolicModel()
trainThis(model,"Symbolic",dataMatchLoss,lambda x: x,lambda o: o,dontViz,trainStates,testStates)
quit()

