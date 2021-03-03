function a = triarea()

addpath(genpath('C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\Utility'))
addpath('C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\src')

options = struct;
options.Lx = 2*pi;  % width of the domain we are integrating over
options.Ly = 2*pi;  % height of the domain we are inegrating over
options.Ky = 4;     % wavenumber of forcing in the y direction
options.Nx = 128;   % spatial grid points along x
options.Ny = 128;   % spatial grid points along y
options.npu = 2^7;  % number of time steps per time unit
npu = options.npu;

load("R40_turbulent_state_k1.mat","s");


endState = s;
endState = endState + 0.3 * randn(128,128,2);

preIntTime = 20;
options.Lt = preIntTime;
domaint = dom.KolmogorovDomainObject(options);
fprintf("Pre-integrating!\n");
traj = int.tangent(domaint,endState);
endState = traj(:,:,:,end);
fprintf("Done pre-integrating!\n");

timeBetweenFrames = 1;
options.Lt = timeBetweenFrames;   % number of (dimensionless) time units to integrate
domaint = dom.KolmogorovDomainObject(options);
frameCount = 8000;
%%
frames = zeros(frameCount,128,128,2);
penultimateFrames = zeros(frameCount,128,128,2);
f = waitbar(0,'Integrating');
for intFrame = 1:frameCount
  waitbar(intFrame/frameCount,f);
  traj = int.tangent(domaint,endState);
  endState = squeeze(traj(:,:,:,end));
  penultimateFrames(intFrame,:,:,:) = squeeze(traj(:,:,:,end-1));
  frames(intFrame,:,:,:) = endState;
end
close(f)

save("endState.mat","endState")
save("biFrames" + frameCount + ".mat","frames","penultimateFrames")

%bigIntegrated = reshape(permute(batches,[2,3,4,1,5]),[128,128,2,batchLength*batchCount]);
%%
psiOuts = zeros(frameCount,128,128);
if false
  for i = 1:frameCount
    psiOuts(i,:,:)=util.fftstream(domaint,squeeze(frames(i,:,:,:)));
  end
end

vortOuts = zeros(frameCount,128,128);
if false
  for i = 1:frameCount
    vortOuts(i,:,:)=util.fftcurl(domaint,squeeze(frames(i,:,:,:)));
  end
end

dissipations=util.fftdissipation(domaint,permute(frames,[2,3,4,1]));
figure
plot(dissipations)
title('Dissipation')

%psiFFTOuts=zeros(Nt,8,8);
%for i = 1:Nt
%  fullFFT=squeeze(fft2(psiOuts(i,:,:)));
%  psiFFTOuts(i,:,:) = fullFFT(1:8,1:8);
%end 
if false
    maxDefects=64;
    featureOuts = zeros(frameCount,maxDefects,2 + 1 + 4)-1;
    chargeCount=zeros([frameCount,1]);
    f = waitbar(0,'Defects');
    parfor defectsFor = 1:frameCount
      waitbar(defectsFor/frameCount,f);
      [locx,locy,locz,h]=defect.reduce(domaint,squeeze(psiOuts(defectsFor,:,:)));
      chargeCount(defectsFor) = numel(locx);
      %hessDet = squeeze(h(1,1,:) .* h(2,2,:) - h(1,2,:) .* h(2,1,:));
      features=[locx, locy, locz,  squeeze(h(1,1,:))  squeeze(h(1,2,:))  squeeze(h(2,1,:))  squeeze(h(2,2,:))];
      padAmount = maxDefects-numel(locx);
      if padAmount < 0
        padAmount = 0;
        features = features(1:maxDefects,:);
        fprintf("Got " + numel(locx) + " defects!!!! at " + defectsFor + "\n")
      end
      featureOuts(defectsFor,:,:)=padarray(features,[padAmount,0],'post');
    end
    close(f)
    figure
    plot(chargeCount)
    title('Charge Count')
end

save("trainingData" + frameCount + ".mat","psiOuts","vortOuts","featureOuts","maxDefects","timeBetweenFrames","npu","dissipations","chargeCount")
fprintf("\nSuccess\n")

