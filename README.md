# BridgeNN
Double Dummy Bridge Implementation using Neural Networks

Goal: 
To create a Neural Network which will be able to accurately predict the final result of a bridge hand given full information.

Final Model: 
Two layers of CNNs with ReLUs, a fully connected layer, and a sigmoid output. The output layer is multipied by the number of cards in the hand.
Inputs are (4, 4, 13) shaped tensors. The player on lead has their cards in the first "column" of the input tensor, and the hands move around the table from there.
Loss was determined by Mean Square Error, and accuracy was measured as a percentage off by zero, one, or two.

Data:
Used DDS Double Dummy Solver (https://github.com/dds-bridge) to get ground truth for full hands.

Results:
Off by 0 Accuracy: 34.2
Off by 1 Accuracy: 80.6
Off by 2 Accuracy: 96.2

Intermediate Process:


Sources Used in Research: 

https://github.com/dds-bridge

http://cs229.stanford.edu/proj2016/report/Mernagh-LearningADoubleDummyBridgeSolver-report.pdf

http://privat.bahnhof.se/wb758135/bridge/index.html

https://link.springer.com/content/pdf/10.1007%2Fb98109.pdf

https://github.com/anntzer/redeal

https://www.youtube.com/watch?v=CRBNI8UdHhE&t=1168s


Time Log:
11/8: Research, Created Google Doc, 1 hour (Jake)
11/10: Research, 90 min (Sam)
11/27: Started writing code for createData, 30 min (Jake)
11/30: Finished the createData for one card hands, 90 min (Jake)
11/30: Learned Something!, created a NN for the play of one card, 3.5 hr (Jake+Sam)
12/5: New CreateData methods, new architecture, tested architecture 3 hr (Jake)
12/7: Attempted transfering to gcloud(Sam), went over previous changes, tried different architectures 3 hr (Sam)
12/10: Tried different encoding of inputs, did some more research, expanded to 2 cards, 4 hrs (Jake + Sam)
12/11: Expanded to 13 cards, fine tuned hyperparameters, finished presentation, 4 hrs (Jake + Sam)
