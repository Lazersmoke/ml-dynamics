# Based on the architecture of: https://arxiv.org/pdf/2008.07515.pdf

import os
# Tell tensorflow not to use the GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Tone down warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("Loading TensorFlow...")
import tensorflow as tf

from tensorflow.keras.layers import Reshape, Dense, Flatten, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model

import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.rc('image', cmap='inferno')

import matplotlib.ticker as mplticker

from scipy import ndimage, misc, signal, stats

### Settings ###
EPOCHS = 1000

batchSize = 128

showDiss = False

printDithers = False
trainDynamics = True
trainLatentModel = False

### /Settings ###

# Import training data from MATLAB file
# Generated by makeStraems.m
fname = "../trainingData8000.mat"
print("Reading " + fname + " ...")
trainingData = sio.loadmat(fname)
examples = tf.cast(trainingData['vortOuts'],tf.float32)

spaceSize = examples.shape[-1]
spaceCenter = spaceSize // 2

data_frameTime = trainingData['timeBetweenFrames'][0,0]
data_npu = trainingData['npu'][0,0]
data_dissipations = trainingData['dissipations']
chargeCount = np.squeeze(trainingData['chargeCount'])

if showDiss:
  plt.plot(data_dissipations)
  plt.show(block=True)

print(
  f'Training data metadata:\n'
  f'  frameTime: {data_frameTime}\n'
  f'  npu: {data_npu}\n'
  f'  numDataPoints: {examples.shape[0]}\n'
  )

forcingWiggle = np.fromfunction(lambda x,y: y + 2 * np.sin(4 * 2 * np.math.pi/128 * x),[128,128])
def vizAutoencoder(original,reconstructed,latent,lab="",epc=""):
  vizLatent(reconstructed,latent,lab,epc)
  vizReal(original,reconstructed,lab,epc)

def vizLatent(reconstructed,latent,lab="",epc=""):
  # Latent plot
  f, (ax1,ax2) = plt.subplots(1, 2,figsize=(20,10))
  ax1.set_title("Vorticity")
  ax1.contourf(reconstructed,cmap="cividis",levels=40)

  ax2.set_title("Latent representation, epoch #" + epc)
  embedDimension = latent.shape[0]
  edgeSize = np.floor(np.sqrt(embedDimension)).astype('int32')
  while (np.mod(embedDimension,edgeSize) > 0):
    edgeSize = edgeSize - 1
  ax2.xaxis.set_major_locator(mplticker.MaxNLocator(integer=True))
  ax2.yaxis.set_major_locator(mplticker.MaxNLocator(integer=True))
  f.colorbar(ax2.imshow(np.reshape(latent,[edgeSize,embedDimension // edgeSize]),cmap="binary"),ax=ax2)

  f.savefig("images/latent" + lab + ".png", bbox_inches='tight')
  plt.close(f)

def vizReal(original,reconstructed,lab="",epc="",direc="images"):
  # Real space plot
  f, (ax1) = plt.subplots(1, 1,figsize=(10,10),dpi=150)
  
  lossAmt = reconstructionLoss(original,reconstructed)
  ax1.set_title("Reconstructed vs Original (" + lab + "), loss = " + str(lossAmt.numpy()) + ", epoch #" + epc)

  ax1.contourf(original,cmap="gray",levels=40)

  ax1.contour(forcingWiggle,colors='green',alpha=0.05,levels=16)
  ax1.contour(reconstructed,cmap="plasma",levels=30)

  f.savefig(direc + "/reconstruction" + lab + ".png", bbox_inches='tight')
  plt.close(f)

  original.numpy().astype('float32').tofile("dumps/original" + lab + '.dat')
  reconstructed.numpy().astype('float32').tofile("dumps/recovered" + lab + '.dat')

print("Data shape: ",examples.shape)

stateDataset = tf.data.Dataset.from_tensor_slices(examples).shuffle(10000).batch(batchSize)
trainStates = stateDataset.shard(2,0)
testStates = stateDataset.shard(2,1)

def makeDynamicPairs(exs):
  return tf.data.Dataset.from_tensor_slices(tf.transpose(tf.convert_to_tensor([exs[:-1],exs[1:]]),perm=[1,0,2]))

class PredictDynamics(Model):
  def __init__(self):
    super(PredictDynamics,self).__init__()
    self.embedDimension = 32
    self.d1 = Dense(32,activation='relu')
    self.d2 = Dense(32,activation='relu')
    self.d3 = Dense(32)

  def call(self,x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return x

class LatentModel(Model):
  def __init__(self):
    super(LatentModel, self).__init__()
    self.inp = tf.keras.layers.InputLayer(input_shape=(batchSize,128,128,1))
    self.mp = MaxPooling2D((2,2))
    self.c1 = Conv2D(32,8,activation='relu')
    self.c2 = Conv2D(32,8,activation='relu')
    self.c3 = Conv2D(32,4,activation='relu')
    self.c4 = Conv2D(64,4,activation='relu')
    self.c5 = Conv2D(128,2,activation='relu')

    self.flat = Flatten()

    self.embedDimension = 32
    self.dIn = Dense(self.embedDimension,activation='relu')
    self.drop = Dropout(0.2)
    self.dOut = Dense(128,activation='relu',input_shape=[self.embedDimension])
    self.rshp = Reshape(target_shape=[4,4,8])

    self.up = UpSampling2D((2,2))
    self.dc5 = Conv2DTranspose(128,(2,2),activation='relu')
    self.dc4 = Conv2DTranspose(64,(4,4),activation='relu')
    self.dc3 = Conv2DTranspose(32,(4,4),activation='relu')
    self.dc2 = Conv2DTranspose(16,(8,8),activation='relu')
    self.dc1 = Conv2DTranspose(1,(8,8))

  def encode(self, x):
    x = tf.expand_dims(x,axis=-1)
    x = self.c1(x)
    x = self.mp(x)

    x = self.c2(x)
    x = self.mp(x)

    x = self.c3(x)
    x = self.mp(x)

    x = self.c4(x)
    x = self.mp(x)

    x = self.c5(x)
    x = self.mp(x)

    x = self.flat(x)
    x = self.dIn(x)

    return x

  def decode(self, x):
    x = self.dOut(x)
    x = self.rshp(x)

    x = self.dc5(x)
    x = self.up(x)

    x = self.dc4(x)
    x = self.up(x)

    x = self.dc3(x)
    x = self.up(x)

    x = self.dc2(x)
    x = self.up(x)

    x = self.dc1(x)
    x = tf.image.crop_to_bounding_box(x,5,5,128, 128)
    return tf.squeeze(x)

  def call(self, x):
    x = self.encode(x)
    x = self.drop(x) # Prevents overfitting
    return self.decode(x)

  def getEigs(self):
    oneHots = tf.eye(self.embedDimension)
    return decode(oneHots)

def dynamicalLoss(delta,prediction):
  errors = delta - prediction
  sqErrs = tf.math.log(tf.math.cosh(errors))
  return tf.math.reduce_mean(sqErrs,axis = -1)

def reconstructionLoss(original,reconstructed):
  errors = original - reconstructed
  sqErrs = tf.math.log(tf.math.cosh(errors))
  return tf.math.reduce_mean(sqErrs,axis = (-1,-2))

def symmetryDither(psi):
  rollAmt = np.random.randint(128)
  psi = tf.roll(psi,rollAmt,axis=-1) # Continuous x-translation

  stepsDiscrete = np.random.randint(8)
  for i in range(stepsDiscrete):
    psi = -tf.reverse(tf.roll(psi,128 // 8,axis = -2),axis=[-1])

  doParity = np.random.randint(2)
  if doParity > 0:
    psi = tf.reverse(psi,axis = [-1,-2])
  return psi

def vizDynamics(ae):
  def f(dynModel,pairs,epoch):
    beforeStates = pairs[:,0,:]
    predictedDiff = dynModel(beforeStates)
    beforeRe = ae.decode(beforeStates)
    afterRe = ae.decode(pairs[:,1,:])
    predictAfterRe = ae.decode(pairs[:,0,:] + predictedDiff)
    n = 8
    for k in range(n):
      if k < len(pairs):
        vizSingleDynamic(beforeRe[k],afterRe[k],predictAfterRe[k],beforeStates[k],pairs[k,1] - pairs[k,0],predictedDiff[k],str(k),str(epoch))
  return f

def plotLatent(f,lz,ax):
  embedDimension = lz.shape[0]
  edgeSize = np.floor(np.sqrt(embedDimension)).astype('int32')
  while (np.mod(embedDimension,edgeSize) > 0):
    edgeSize = edgeSize - 1
  ax.xaxis.set_major_locator(mplticker.MaxNLocator(integer=True))
  ax.yaxis.set_major_locator(mplticker.MaxNLocator(integer=True))
  f.colorbar(ax.imshow(np.reshape(lz,[edgeSize,embedDimension // edgeSize]),cmap="binary"),ax=ax)

def vizSingleDynamic(beforeRe,afterRe,predictedRe,before,trueDiff,predictedDiff,lab="",epc=""):

  f, ((axr1,axr2,axr3),(axl1,axl2,axl3)) = plt.subplots(2, 3,figsize=(20,20))
  axr1.set_title("Before (" + lab + ")")
  axr1.contourf(beforeRe,cmap="cividis",levels=40)
  axl1.set_title("Before (" + lab + "), latent representation")
  plotLatent(f,before,axl1)

  axr2.set_title("True After (" + lab + ")")
  axr2.contourf(afterRe,cmap="cividis",levels=40)
  axl2.set_title("True diff (" + lab + "), latent representation")
  plotLatent(f,trueDiff,axl2)

  axr3.set_title("Predicted After (" + lab + "), epoch#" + epc)
  axr3.contourf(predictedRe,cmap="cividis",levels=40)
  axl3.set_title("Predictied diff (" + lab + "), latent representation")
  plotLatent(f,predictedDiff,axl3)

  f.savefig("dynamicsImages/dynamic" + lab + ".png", bbox_inches='tight')
  plt.close(f)

def vizLatentModel(model,test_orig,epoch):
  symOrig = symmetryDither(test_orig)
  recovered = model(symOrig,training=False)
  latent = model.encode(symOrig)
  oneHots = tf.eye(model.embedDimension)
  eigs = model.decode(oneHots)
  n = 8
  for l in range(eigs.shape[0]):
    vizLatent(eigs[l],oneHots[l],"Eigen " + str(l),str(epoch))
  for k in range(n):
    # If this is a truncated epoch, don't crash
    if k < len(test_orig):
      # Save the weights every time we visualize anything, so we can always get that network back
      vizAutoencoder(symOrig[k],recovered[k],latent[k],str(k),str(epoch))

if printDithers:
  testOmega = examples[np.random.randint(examples.shape[0])]
  for i in range(20):
    dithered = symmetryDither(testOmega)
    vizReal(dithered,dithered,"dither" + str(i),"-",direc="dither")

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

    # How many examples to output as graphs every vizEpochs
    n = 8
    vizEpochs = 50
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
      f'Reconstruction Loss: {train_loss.result()}, '
      f'Test Reconstruction Loss: {test_loss.result()}'
    )

  # Still save weights even if n=0 graphs
  model.save_weights(checkpoint_path.format(epoch=EPOCHS))

if trainDynamics:
  ae = LatentModel()
  ae.load_weights(tf.train.latest_checkpoint("training_data_AE"))
  exBatches = tf.unstack(tf.split(examples,100))
  print("Encoding...")
  encExamples = tf.reshape([ae.encode(x) for x in exBatches],[examples.shape[0],ae.embedDimension])

  dynamicsModel = PredictDynamics()
  dynPairs = makeDynamicPairs(encExamples).shuffle(10000).batch(batchSize)
  trainPairs = dynPairs.shard(2,0)
  testPairs = dynPairs.shard(2,1)
  def getBefore(x):
    return x[:,0]
  def getDiff(x):
    return x[:,1] - x[:,0]
  trainThis(dynamicsModel,"dynamics",dynamicalLoss,getBefore,getDiff,vizDynamics(ae),trainPairs,testPairs)
  quit()

if trainLatentModel:
  autoencoder = LatentModel()
  trainThis(autoencoder,"AE",reconstructionLoss,symmetryDither,lambda o: o,vizLatentModel,trainStates,testStates)
  quit()

