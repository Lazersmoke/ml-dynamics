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

batchSize = 16

exactDefects = 12

sortLossCoeff = 0.1

powerFitCount = 5

doSparse = True

### /Settings ###

def intGenerator():
  print("Starting MATLAB...")
  eng = matlab.engine.start_matlab()

  eng.addpath(eng.genpath(r'C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\Utility'))
  eng.addpath(r'C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\src')
  eng.workspace['domain'] = eng.getDomain()
  #eng.workspace['endState'] = eng.load("R40_turbulent_state_k1.mat","s")['s']
  
  k = 0
  while True:
    print("Leap " + str(k))
    k = k + 1
    yield eng.eval("leapIntegrate(0.01,2," + str(exactDefects) + ",domain);")

  #while True:
    #defectDatums = []
    #while True:
      #print("Integrating...")
      #defectData = eng.eval("getDefects(domain,endState)")
      #if len(defectData) == exactDefects:
        #defectDatums.append(defectData)
        #eng.workspace['endState'] = eng.eval("stepIntegrate(0.01,domain,endState)")
      #else:
        #print("Skipping region with (" + str(len(defectData)) + ") defects...")
        #eng.workspace['endState'] = eng.eval("stepIntegrate(2,domain,endState)")
        #break
    #if len(defectDatums) > 0:
      #yield defectDatums

timeSliceSize = 3
fromMatlabDataset = (tf.data.Dataset
  .from_generator(intGenerator,output_signature = tf.TensorSpec(shape=(None,exactDefects,7), dtype=tf.float32))
  .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x).batch(timeSliceSize,drop_remainder=True))
  .prefetch(10)
  .take(50000)
  .cache(filename='training_data_DefectData/defects50k'))

def squishToTwoPi(x):
  return tf.math.floormod(x[...,:2],2 * np.math.pi)

# Expects defects in second to last dimension
def sortDefects(left,right):
  leftExpand = tf.repeat(tf.expand_dims(left,-2),exactDefects,axis=-2)
  diffs = squishToTwoPi(tf.abs(leftExpand - tf.expand_dims(right,-3)))
  nearness = tf.math.reduce_sum(diffs * diffs,-1)
  argmins = tf.argmin(nearness,1)
  fixedRight = tf.gather(right,argmins)
  #debugSorter = True
  #if debugSorter:
    #maxOrigDiff = tf.math.reduce_max(tf.math.reduce_sum((left[...,:2] - right[...,:2]) * (left[...,:2] - right[...,:2]),-1),-1)
    #if (maxOrigDiff > 0.2) or (tf.math.reduce_sum(tf.abs(argmins - np.arange(exactDefects)),-1).numpy() > 0):
      #print("Left",left[...,:3])
      #print("Right",right[...,:3])
      #print("nearness",nearness)
      #print("argmins",argmins)
      #print("fixedRight",fixedRight[...,:3])
      #print("maxOrigDiff",maxOrigDiff)
  return fixedRight

def sortAlongDefects(xsTensor):
  #xs = tf.unstack(xsTensor)
  return tf.scan(sortDefects, xsTensor)
  #out = []
  #out.append(xs[0])
  #for i in range(len(xs)-1):
    #out.append(sortDefects(out[i],xs[i+1]))
  #return tf.stack(out,0)

testTrajDataset = (tf.data.Dataset
  .from_generator(intGenerator,output_signature = tf.TensorSpec(shape=(None,exactDefects,7), dtype=tf.float32))
  .take(10)
  .cache(filename='training_data_DefectData/defectTestTrajectories10')
  .map(sortAlongDefects)
  .shuffle(100))

#for traj in list(testTrajDataset.as_numpy_iterator()):
  #print(traj.shape)

#print(fromMatlabDataset)
#print(list(fromMatlabDataset.map(lambda x: tf.math.count_nonzero(x[:,:,0])).as_numpy_iterator()))
#print(list(fromMatlabDataset.map(lambda x: x[:,0,0]).as_numpy_iterator()))
#print(fromMatlabDataset)

positionKern = [0,1,0.0]
timeDifferenceKern = [-0.5,0,0.5]
timeSecondDifferenceKern = [1,-2,1.0]

def safePow(a,b):
  return tf.math.sign(a) * tf.math.pow(tf.math.maximum(tf.abs(a),0.00000000000001),b)

def buildTerm(x,exponents):
  timeKernels = tf.stack([positionKern,timeDifferenceKern,timeSecondDifferenceKern],axis=0)
  # x is [batchSize,timeSliceSize,exactDefects,vars]
  # timeKernels is [orders,timeSliceSize]
  timeDerivs = tf.tensordot(x,timeKernels,axes=[1,1])
  # timeDerivs is [batchSize,exactDefects,vars,orders]
  # exponents is [vars,orders]
  exponents = tf.expand_dims(tf.expand_dims(exponents,axis=0),axis=0)
  term = tf.math.reduce_prod(safePow(timeDerivs,exponents),axis=[-1,-2])
  # term is [batchSize,exactDefects]
  return term

def termMatrix(varCount,opCount,v,o):
  m = tf.one_hot(varCount * v + o,varCount * opCount,dtype=tf.float32)
  m = tf.reshape(m,[varCount,opCount])
  return m

def takeDotProducts(vecs):
  n = len(vecs)
  upperTriangle = []
  for i in range(n):
    for j in range(i,n):
      upperTriangle.append(tf.einsum("...i,...i->...",vecs[i],vecs[j]))
  return upperTriangle

def getKern(dns):
  return tf.squeeze(dns(tf.constant(1.0,shape=[1,1])))

stateDataset = fromMatlabDataset.map(sortAlongDefects).batch(batchSize).cache("training_data_DefectData/batchedAndSorted50k").shuffle(10000)
trainStates = stateDataset.shard(2,0)
testStates = stateDataset.shard(2,1)

class SymbolicModel(Model):
  def __init__(self):
    super(SymbolicModel,self).__init__()
    self.varCount = 7
    self.timeOrder = 2 + 1
    self.composite = Dense(3,input_dim=1,use_bias=False)
    self.diffNormExps = Dense(powerFitCount,input_dim=1,use_bias=False)
    self.diffNormExpsHess = Dense(powerFitCount,input_dim=1,use_bias=False)
    self.diffsCoupling = Dense(powerFitCount,input_dim=1,use_bias=False)
    self.hessCoupling = Dense(powerFitCount,input_dim=1,use_bias=False)

    self.compositeHess = Dense(2,input_dim=1,use_bias=False)
    self.chargeShareExps = Dense(powerFitCount,input_dim=1,use_bias=False)
    self.chargeShareCoupling = Dense(powerFitCount,input_dim=1,use_bias=False)
    #self.extraTermStorage = Dense(4*self.varCount*self.timeOrder,input_dim=1,use_bias=False,name='extraTerms')

  def timeIntegrate(self,defectState):
    eqn = self.getEqn(defectState)[0]
    comp = getKern(self.composite)
    compositeMinusTime = comp - tf.one_hot(1,3) * comp
    timeDeriv = tf.math.reduce_sum(tf.math.divide_no_nan(tf.einsum("i,...i->...",compositeMinusTime,eqn),comp[0]),axis=2)
    return tf.squeeze(timeDeriv)

  def timeIntegrateHess(self,defectState):
    eqn = self.getEqn(defectState)[1]
    comp = getKern(self.compositeHess)
    compositeMinusTime = comp - tf.one_hot(1,2) * comp
    timeDeriv = tf.math.reduce_sum(tf.math.divide_no_nan(tf.einsum("i,...i->...",compositeMinusTime,eqn),comp[0]),axis=2)
    return tf.squeeze(timeDeriv)

  def getTrueDQDT(self,x):
    dxdt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,0,1)) # dx/dt
    dydt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,1,1)) # dy/dt

    dqdt = tf.expand_dims(tf.stack([dxdt,dydt],axis=-1),-2)
    return tf.squeeze(dqdt)

  def getEqn(self,x):

    lossAmt = 0

    hess11dt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,3,1))
    hess12dt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,4,1))
    hess21dt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,5,1))
    hess22dt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,6,1))

    dHdt = tf.stack([tf.stack([hess11dt,hess12dt],-1),tf.stack([hess21dt,hess22dt],-1)],-1)

    hess11 = buildTerm(x,termMatrix(self.varCount,self.timeOrder,3,0))
    hess12 = buildTerm(x,termMatrix(self.varCount,self.timeOrder,4,0))
    hess21 = buildTerm(x,termMatrix(self.varCount,self.timeOrder,5,0))
    hess22 = buildTerm(x,termMatrix(self.varCount,self.timeOrder,6,0))

    # 11 12
    # 21 22
    hessMat = tf.stack([tf.stack([hess11,hess12],-1),tf.stack([hess21,hess22],-1)],-1)

    locx = buildTerm(x,termMatrix(self.varCount,self.timeOrder,0,0))
    diffsX = tf.tile(tf.expand_dims(locx,axis=-1),[1,1,locx.shape[1]])
    diffsX = diffsX - tf.transpose(diffsX,perm=[0,2,1])

    locy = buildTerm(x,termMatrix(self.varCount,self.timeOrder,1,0))
    diffsY = tf.tile(tf.expand_dims(locy,axis=-1),[1,1,locy.shape[1]])
    diffsY = diffsY - tf.transpose(diffsY,perm=[0,2,1])

    diffsVec = tf.stack([diffsX,diffsY],axis=-1)
    diffNorms = tf.einsum("...i,...i->...",diffsVec,diffsVec)
    poweredNorms = safePow(tf.expand_dims(diffNorms,axis=-1),getKern(self.diffNormExps))
    # Index labelling:
    # Us, Other, Jcoordinate, Icoupling
    directCouplingTerm = tf.einsum("i,...uoi,...uoj->...uoj",getKern(self.diffsCoupling),poweredNorms,diffsVec)

    hessDiffsVec = tf.einsum("...uij,...uoj->...uoi",hessMat,diffsVec)
    hessPoweredNorms = safePow(tf.expand_dims(diffNorms,axis=-1),getKern(self.diffNormExpsHess))
    hessCouplingTerm = tf.einsum("i,...uoi,...uoj->...uoj",getKern(self.hessCoupling),hessPoweredNorms,hessDiffsVec)

    dxdt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,0,1)) # dx/dt
    dydt = buildTerm(x,termMatrix(self.varCount,self.timeOrder,1,1)) # dy/dt

    dqdt = tf.broadcast_to(tf.expand_dims(tf.stack([dxdt,dydt],axis=-1),-2),diffsVec.shape)

    eqn = tf.stack([dqdt,directCouplingTerm,hessCouplingTerm],-1)

    chargeSharingPoweredNorms = safePow(tf.expand_dims(diffNorms,axis=-1),getKern(self.chargeShareExps))
    chargeSharingTerm = tf.einsum("i,...uoi,...ojk->...ujk",getKern(self.chargeShareCoupling),chargeSharingPoweredNorms,hessMat)

    #biTensorPoweredNorms = safePow(tf.expand_dims(diffNorms,axis=-1),getKern(self.biTensorExps))
    #biTensorTerm = tf.einsum("...l,...ljk->...jk",tf.einsum("...i,...i->...",getKern(self.biTensorCoupling),biTensorPoweredNorms),diffsVec)
    #print("hess mat",hessMat.shape)
    #print("Charge share term",chargeSharingTerm.shape)

    eqnH = tf.stack([dHdt,chargeSharingTerm],-1)
    return [eqn,eqnH]

  def call(self,x):
    eqns = self.getEqn(x)
    eqnq = eqns[0]
    eqnH = eqns[1]

    residualq = tf.einsum("i,...i->...",getKern(self.composite),eqnq)
    residualH = tf.einsum("i,...i->...",getKern(self.compositeHess),eqnH)

    # Sum over all charge contributions, then abs, then sum over x and y directions
    lossAmt = tf.math.reduce_sum(tf.abs(tf.math.reduce_sum(residualq,-2)),-1)

    # Sum over abs(matrix elements)
    lossAmt += tf.math.reduce_sum(tf.abs(residualH),[-1,-2])

    # Time derivative normalization loss
    for c in [self.composite,self.compositeHess]:
      lossAmt += 10. * tf.abs(1.0 - getKern(c)[0])

    for exp in [self.diffNormExps,self.diffNormExpsHess,self.chargeShareExps]:
      # Unphysical exponent loss
      lossAmt += 100. * tf.math.reduce_sum(tf.math.maximum(0.,getKern(exp)),axis=-1)
      # Unsorted exponent loss
      lossAmt += 100. * tf.math.reduce_sum(tf.math.maximum(0.,sortLossCoeff-tf.experimental.numpy.diff(getKern(exp))),axis=-1)

    for coup in [self.diffsCoupling,self.hessCoupling,self.chargeShareCoupling]:
      # Coupling normalization loss
      lossAmt += tf.abs(1. - tf.math.reduce_sum(tf.abs(getKern(coup))))
      if doSparse:
        # Coupling sparsification loss
        lossAmt += tf.math.reduce_sum(0.8 * tf.math.sin(np.math.pi * tf.math.minimum(1.,5. * tf.abs(getKern(coup)))),-1)

    return lossAmt

  def printAuxLoss(self):
    for n,c in [("diff",self.composite),("chrg",self.compositeHess)]:
      print("Time derivative normalization loss (" + n + ")",tf.abs(1.0 - getKern(c)[0]).numpy())

    for n,e in [("diff",self.diffNormExps),("hess",self.diffNormExpsHess),("chrg",self.chargeShareExps)]:
      print("Unphysical exponent loss (" + n + ") ",tf.math.reduce_sum(tf.math.maximum(0.,getKern(e)),axis=-1).numpy())

      print("Unsorted exponent loss (" + n + ") ",tf.math.reduce_sum(tf.math.maximum(0.,sortLossCoeff-tf.experimental.numpy.diff(getKern(e))),axis=-1).numpy())

    for n,coup in [("diff",self.diffsCoupling),("hess",self.hessCoupling),("chrg",self.chargeShareCoupling)]:
      print("Coupling normalization loss (" + n + ") ",tf.abs(1. - tf.math.reduce_sum(tf.abs(getKern(coup)))).numpy())
      if doSparse:
        sparseLoss = tf.math.reduce_sum(0.8 * tf.math.sin(np.math.pi * tf.math.minimum(1.,5. * tf.abs(getKern(coup)))),-1)
        print("Coupling sparsification loss (" + n + ") ",sparseLoss.numpy())

  def explainYourself(self):
    strOut = "\n{:.3e} dqdt\n{:.3e}[".format(getKern(self.composite)[0].numpy(),getKern(self.composite)[1].numpy())
    for i in range(getKern(self.diffNormExps).shape[0]):
      strOut += " {:+.3e} (dq) r^2({:.3e})".format(getKern(self.diffsCoupling)[i].numpy(),getKern(self.diffNormExps)[i].numpy())
    strOut += "]\n{:.3e}[".format(getKern(self.composite)[2].numpy())
    for i in range(getKern(self.diffNormExpsHess).shape[0]):
      strOut += " {:+.3e} (H dq) r^2({:.3e})".format(getKern(self.hessCoupling)[i].numpy(),getKern(self.diffNormExpsHess)[i].numpy())
    strOut += "]\n\n{:.3e} dHdt\n{:.3e}[".format(getKern(self.compositeHess)[0].numpy(),getKern(self.compositeHess)[1].numpy())
    for i in range(getKern(self.chargeShareExps).shape[0]):
      strOut += " {:+.3e} (Hl) r^2({:.3e})".format(getKern(self.chargeShareCoupling)[i].numpy(),getKern(self.chargeShareExps)[i].numpy())
    strOut += "]\n\n" + self.texport()
    return strOut

  def texport(self):
    tex = "\\begin{align*}"
    tex += "0&= {:+.2f} \\frac{{dq_i}}{{dt}} {:+.4f} (\\Delta q)_{{ij}}\\left[".format(getKern(self.composite)[0].numpy(),getKern(self.composite)[1].numpy())
    for i in range(getKern(self.diffNormExps).shape[0]):
      tex += " {:+.2f} \\frac{{1}}{{(r_{{ij}}^2)^{{{:.2f}}}}} ".format(getKern(self.diffsCoupling)[i].numpy(),-getKern(self.diffNormExps)[i].numpy())
    tex += "\\right] {:+.4f}H_{{jk}}(\\Delta q)_{{ik}}\\left[".format(getKern(self.composite)[2].numpy())
    for i in range(getKern(self.diffNormExpsHess).shape[0]):
      tex += " {:+.2f} \\frac{{1}}{{(r_{{ij}}^2)^{{{:.2f}}}}} ".format(getKern(self.hessCoupling)[i].numpy(),-getKern(self.diffNormExpsHess)[i].numpy())
    tex += "\\right]\\\\ 0&= {:+.2f} \\frac{{dH^k_{{ij}}}}{{dt}} {:+.4f}H^l_{{ij}}\\left[".format(getKern(self.compositeHess)[0].numpy(),getKern(self.compositeHess)[1].numpy())
    for i in range(getKern(self.chargeShareExps).shape[0]):
      tex += " {:+.2f} \\frac{{1}}{{(r_{{kl}}^2)^{{{:.2f}}}}} ".format(getKern(self.chargeShareCoupling)[i].numpy(),-getKern(self.chargeShareExps)[i].numpy())
    tex += "\\right]\\end{align*}"
    return tex

def dataMatchLoss(x,residuals):
  return tf.keras.losses.log_cosh(0,residuals)

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

  trainTicks = 0
  for x in trainSet:
    trainTicks += 1

  testTicks = 0
  for x in testSet:
    testTicks += 1

  cs = ["'","\b`","\b\"","\b\\","\b=","\b-","\b,","\b."]
  csb = [".","\b,","\b-","\b=","\b/","\b\"","\b`","\b'"]
  csn = 8

  bst = "[" + (1 + trainTicks // csn) * " " + "]" + (2 + trainTicks // csn) * "\b"
  bstb = "[" + (1 + testTicks // csn) * " " + "]" + (2 + testTicks // csn) * "\b"

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

    k = 0
    print(bst,end="",flush=True)
    for original in trainSet:
      # Progess bar
      print(cs[k],end="",flush=True)
      k = (k + 1) % csn
      train_step(original)

    print()
    print(bstb,end="",flush=True)
    k = 0
    for test_orig in testSet:
      print(csb[k],end="",flush=True)
      k = (k + 1) % csn
      test_step(test_orig)

    vizEpochs = 15
    # Also vizualize on last epoch
    if ((epoch % vizEpochs == 0) or (epoch == EPOCHS - 1)):
      print("\b!",end="",flush=True)
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
  if False:
    compareModelWithIntegrator(x)
  #print(y[0])
  x(y)[0]
  print(x.explainYourself())
  x.printAuxLoss()
  #for i in range(len(x.layers)):
    #print(x.layers[i].get_weights()[0])
  return

def compareModelWithIntegrator(mod):
  trueTrajectory = sortAlongDefects(list(testTrajDataset.take(1).as_numpy_iterator())[0])
  trueTrajectoryRed = trueTrajectory[:-2]
  trueTrajectoryFrames = tf.unstack(trueTrajectory)
  numFrames = len(trueTrajectoryFrames) - 2
  predTangents = np.zeros([numFrames,exactDefects,2])
  trueTangents = np.zeros([numFrames,exactDefects,2])
  for i in range(numFrames):
    curTimeSlice = tf.stack(trueTrajectoryFrames[i:3+i],0)
    dqdt = mod.timeIntegrate(tf.expand_dims(curTimeSlice,0))
    predTangents[i] = dqdt
    truedqdt = mod.getTrueDQDT(tf.expand_dims(curTimeSlice,0))
    trueTangents[i] = truedqdt
    #print("Predicted ", dqdt)
    #print("True ", truedqdt)
    #knownPosition = knownPosition + tf.pad(tf.squeeze(dqdt * 0.01),[[0,0],[0,5]])
  print("Frames: ",numFrames)
  for c in range(exactDefects):
    plt.plot(trueTrajectoryRed[:,c,0],trueTrajectoryRed[:,c,1],'b')
  tangentScale = 5.
  for c in range(exactDefects):
    plt.scatter(tangentScale * trueTangents[:,c,0] + trueTrajectoryRed[:,c,0],tangentScale * trueTangents[:,c,1] + trueTrajectoryRed[:,c,1],marker='x',c='red')
    plt.scatter(tangentScale * predTangents[:,c,0] + trueTrajectoryRed[:,c,0],tangentScale * predTangents[:,c,1] + trueTrajectoryRed[:,c,1],marker='x',c='green')
  plt.xlim([0,2 * np.math.pi])
  plt.ylim([0,2 * np.math.pi])
  plt.show(block=True)

model = SymbolicModel()
trainThis(model,"Symbolic",dataMatchLoss,lambda x: x,lambda o: o,dontViz,trainStates,testStates)
quit()

