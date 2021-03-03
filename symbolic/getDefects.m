function defectData = getDefects(domain,endState)

[locx,locy,locz,h] = defect.reduce(domain,util.fftstream(domain,endState));
%chargeCount(defectsFor) = numel(locx);
%hessDet = squeeze(h(1,1,:) .* h(2,2,:) - h(1,2,:) .* h(2,1,:));
defectData=[locx, locy, locz,  squeeze(h(1,1,:))  squeeze(h(1,2,:))  squeeze(h(2,1,:))  squeeze(h(2,2,:))];
