function net_BGD=CreateFeedforwardNN(TrainInputs,netsize)
      
   net_BGD = newff(minmax(TrainInputs),netsize,{'logsig','logsig'},'trainlm');
   
   % Parameters setting
   net_BGD.trainparam.goal = 10e-10;    
   net_BGD.trainparam.show = 50; 

   net_BGD.trainparam.epochs = 1000;   

end