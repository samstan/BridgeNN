# BridgeNN
Double Dummy Bridge Implementation using Neural Networks

(Google Presentation: https://docs.google.com/presentation/d/1tSIXhCBVx2m3kigLrA73syW74sJSr-ld80KX8RmpEsA/edit#slide=id.g49dcd0f2da_0_91)

### Goal: 
To create a Neural Network which will be able to accurately predict the final result of a bridge hand given full information.

### Final Model: 
Two layers of CNNs with ReLUs, a fully connected layer, and a sigmoid output. The output layer is multipied by the number of cards in the hand.
Inputs are (4, 4, 13) shaped tensors. The player on lead has their cards in the first "column" of the input tensor, and the hands move around the table from there.
Loss was determined by Mean Square Error, and accuracy was measured as a percentage off by zero, one, or two.

### Data:
Used DDS Double Dummy Solver (https://github.com/dds-bridge) to get ground truth for full hands. Ground truth saved in file 'sol100000.txt', then parsed into tensors to run through nets. 

### Results:

Off by 0 Accuracy: 34.2

Off by 1 Accuracy: 80.6

Off by 2 Accuracy: 96.2

### Intermediate Process:
#### First Step: Learn Something
To start, we trained a network to predict the result of a one-card ending. We began using (1, 218) sized tensors which indicated the trump, suit, and cards held by each player. We used three fully-connected layers and a sigmoid output. This got us a reasonable accuracy, but was overfitting. We decided to switch to an architecture with convolutional layers, so we needed to adjust the input type.

#### Second Go Around
We changed the input tensor format to (5,4,13) for lead and individual hands. We assumed that we were always in NT to simplify the problem. We introduced the new architecture: 
* 13 convolutions of 3x3 with 1 stride, 1 padding 
* ReLU
* MaxPool 2x2, stride 2
* Fully connected layer to softmax

This new architecture performed worse accuracy wise, but was not overfitting as greatly. It was also a much smaller architecture, parameters wise, so we decided to stick with this style of model. One of the concerns with the first step was that to get good accuracy, we required a large number of parameters, despite the simplicity of the initial problem. That architecture seemed as though it would scale poorly, while this one gave us a better shot. One thing that we tried for a bit was to see which player was winning the trick, which is why this architecture has a softmax rather than sigmoid at the final layer. This was removed in favor of choosing the team winning the trick for later architectures.

#### Incremental Updates
We decided to adjust the way we thought about who was on lead by rotating the inputs to have the person on lead first, which reduced the tensor format to (4,4,13). We also removed the MaxPool layer, partially due to the work of Dali (2018). Our new architecture was the final one we settled on. We tested it on one card and two cards with approximately 70% off by 0 accuracy for both, and 96% off by 1 accuracy for two cards. Thus, we determined that this would be our final architecture. The result for the final architecture on all 13 cards (as described above) were fairly similar to a 2009 paper by Mossakowski and Mandziuk. 

#### Ground Truth/Data for 1 and 2 Cards
While we used the data from DDS for 13 cards, they did not have data for fewer cards, so we made our own. To do so, we randomly choose the requisite number of values between 1 and 52 (without replacement). These values were converted into a location in a (4,13) tensor to represent a player's hand. Then we generated the ground truth by simulating the play of the cards. For one card, it was simple enough to simply see which card was the highest in the suit held by player 0. For two cards, we simulate all 16 combinations of play, determine which are legal, and then determine the optimal play given the desire from both sides to maximize their trick taking. This brute force method was used to generate the training and validation data for 1cardNN and 2cardNN.


### Sources Used in Research: 

https://github.com/dds-bridge

http://cs229.stanford.edu/proj2016/report/Mernagh-LearningADoubleDummyBridgeSolver-report.pdf

http://privat.bahnhof.se/wb758135/bridge/index.html

https://link.springer.com/content/pdf/10.1007%2Fb98109.pdf

https://github.com/anntzer/redeal

https://www.youtube.com/watch?v=CRBNI8UdHhE&t=1168s

https://link.springer.com/chapter/10.1007%2F978-3-540-24844-6_142 


### Time Log:
11/8: Research, Created Google Doc, 1 hour (Jake)

11/10: Research, 90 min (Sam)

11/27: Started writing code for createData, 30 min (Jake)

11/30: Finished the createData for one card hands, 90 min (Jake)

11/30: Learned Something!, created a NN for the play of one card, 3.5 hr (Jake+Sam)

12/5: New CreateData methods, new architecture, tested architecture 3 hr (Jake)

12/7: Attempted transfering to gcloud(Sam), went over previous changes, tried different architectures 3 hr (Sam)

12/10: Tried different encoding of inputs, did some more research, expanded to 2 cards, 4 hrs (Jake + Sam)

12/11: Expanded to 13 cards, fine tuned hyperparameters, finished presentation, 4 hrs (Jake + Sam)
