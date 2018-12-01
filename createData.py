#Code to create data sets for training

import torch
import random
import os

SEED = 955 #Seed provided in order to make sure the same data is generated each time.
TRAINING_SIZE = 100000
VALIDATION_SIZE = 10000

def convertToTensor(inputs, result):
    temp = torch.cat((inputs[0], inputs[1], inputs[2][0], inputs[3][0], inputs[4][0], inputs[5][0]),0)
    return torch.cat((temp.float(), torch.tensor([result]).float()),0)
    

def generateData():
    """Creates a file with randomly generated data that 
    has both the inputs and outputs."""
    random.seed(SEED)
    for i in range(TRAINING_SIZE):
        inputs = getRandomDeal(1)
        result = getResultOneCard(inputs)
        torch.save(convertToTensor(inputs,result), os.path.join('trn', str(i)+'.pt'))
    for i in range(VALIDATION_SIZE):
        inputs = getRandomDeal(1)
        result = getResultOneCard(inputs)
        torch.save(convertToTensor(inputs,result), os.path.join('val', str(i)+'.pt'))

def getResultOneCard(inputs):
    """Generates the ground truth for a hand with only card. Inputs is a list in 
    the order [trump, lead, north, east, south, west]. Ground truth is defined by 
    a 0 for EW and a 1 for NS""" 
    cards = [getCardAsTwoInputs(inputs[2][0]), getCardAsTwoInputs(inputs[3][0]), getCardAsTwoInputs(inputs[4][0]), getCardAsTwoInputs(inputs[5][0])]
    lead = getValue(inputs[1])
    topCard = [lead, cards[lead]]
    for i in range(3):
        lead += 1
        if lead > 3:
            lead = 0
        if cards[lead][0] == getValue(inputs[0]) or cards[lead][0] == topCard[1][0]:
            if cards[lead][1] > topCard[1][1]:
                topCard = [lead, cards[lead]]
            elif cards[lead][0] == getValue(inputs[0]):
                if not topCard[1][0] == getValue(inputs[0]):
                    topCard = [lead, cards[lead]]
    return (1 - topCard[0]%2)

def getRandomDeal(cards):
    """Creates a random deal, determining the trump and who is on lead. 
    The input cards determines how many cards are dealt to each person ([1,13]).
    Returns a list of 6 one-hot tensors:  
    trump: 1x5 (NT, S, H, D, C)
    lead: 1x4 (N, E, S, W)
    north: cardsx52 (AS, ..., 2C)
    east: cardsx52 (AS, ..., 2C)
    south: cardsx52 (AS, ..., 2C)
    west: cardsx52 (AS, ..., 2C)"""
    trump = torch.zeros(5)
    lead = torch.zeros(4)
    north = torch.zeros(cards, 52)
    east = torch.zeros(cards, 52)
    south = torch.zeros(cards, 52)
    west = torch.zeros(cards, 52)
    trump[random.randint(0, 4)] = 1
    lead[random.randint(0, 3)] = 1
    cardInts = []
    while(len(cardInts) < cards*4):
        c = random.randint(0, 51)
        if not c in cardInts:
            cardInts += [c]
    i4 = 0
    for i in range(0, cards*4):
        if i%4 == 0:
            north[i4][cardInts[i]] = 1
        elif i%4 == 1:
            east[i4][cardInts[i]] = 1
        elif i%4 == 2:
            south[i4][cardInts[i]] = 1
        else:
            west[i4][cardInts[i]] = 1
            i4 += 1
    return [trump, lead, north, east, south, west]

def getCardAsTwoInputs(card):
    """Takes a one-hot encoding of a card and returns a list describing the card
    by its suit (1=Spade, ..., 4=Club) and its value (0 = 2, ..., 12 = A)"""
    val = getValue(card)
    return [(int)(val/13) + 1, 12-(val%13)]

def getValue(onehot):
    """Takes a one-hot encoding and returns the value of the 1 (the position)"""
    val = 0
    for i in range(len(onehot)):
        if(onehot.data[i] == 1):
            val = i
    return val