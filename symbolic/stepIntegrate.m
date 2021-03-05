function endState = stepIntegrate(stepSize,domain,startState)
domain.Lt = stepSize;
domain.Nt = max(ceil(domain.Lt*domain.npu),4);

traj = int.tangent(domain,startState);
endState = squeeze(traj(:,:,:,end));
