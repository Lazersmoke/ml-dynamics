function defectDatums = leapIntegrate(stepSize,leapSize,exactDefects,domain)
global endState
%endState
while 1
  defectDatums = [];
  while 1
    %print("Integrating...")
    defectData = getDefects(domain,endState);
    if length(defectData) == exactDefects
      defectDatums = cat(1,defectDatums,reshape(defectData,1,exactDefects,7));
      %size(defectDatums,1)
      endState = stepIntegrate(stepSize,domain,endState);
    else
      %print("Skipping region with (" + str(len(defectData)) + ") defects...")
      endState = stepIntegrate(leapSize,domain,endState);
      break
    end
  end
  if size(defectDatums,1) > 0
    break
  end
end
