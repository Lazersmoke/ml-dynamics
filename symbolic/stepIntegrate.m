function endState = stepIntegrate(domain,startState)

traj = int.tangent(domain,startState);
endState = squeeze(traj(:,:,:,end));
