# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:23:21 2020

@author: haozh
"""

#!/usr/bin/env python
import re, random, math, collections, itertools
import numpy as np
random.seed(729)
PRINT_ERRORS=0

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = re.findall(r"[a-z\-]+", posDictionary.read())

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = re.findall(r"[a-z\-]+", negDictionary.read())

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

  #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"








# def testDictionarynew(sentencesTest, dataName, sentimentDictionary, threshold):
#     total=0
#     correct=0
#     totalpos=0
#     totalneg=0
#     totalpospred=0
#     totalnegpred=0
#     correctpos=0
#     correctneg=0
    
#     negation_words = ['not', 'no']
#     # negation_words = ["not"]
    
#     for sentence, sentiment in sentencesTest.items():
#         Words = re.findall(r"[\w']+", sentence)
#         score = 0
#         current = 0
#         for word in Words:
#             if word in sentimentDictionary:
#                 if np.any(np.in1d(Words[current - 2 : current], negation_words)):
#                     score+=sentimentDictionary[word]*(-1)
#                     current += 1
#                 else:
#                     score+=sentimentDictionary[word]
#                     current += 1            
#             else:
#                 current += 1

#         total+=1
#         if sentiment=="positive":
#             totalpos+=1
#             if score>=threshold:
#                 correct+=1
#                 correctpos+=1
#                 totalpospred+=1
#             else:
#                 correct+=0
#                 totalnegpred+=1
#         else:
#             totalneg+=1
#             if score<threshold:
#                 correct+=1
#                 correctneg+=1
#                 totalnegpred+=1
#             else:
#                 correct+=0
#                 totalpospred+=1
                
#     print("STEP 5", dataName)
#     print("total:", total)                #总数
#     print("correct:", correct)            #正确的数量
#     print("totalpos:", totalpos)          #pos的数量  
#     print("totalpospred:", totalpospred)  #预测是pos的数量
#     print("totalneg:", totalneg)          #neg的数量
#     print("totalnegpred:", totalnegpred)  #预测是neg的数量
#     print("correctpos:", correctpos)      #正确被预测为pos的数量
#     print("correctneg:", correctneg)      #正确被预测为neg的数量
    
#     Accuracy = correct/total
    
#     Precision_Pos = correctpos/totalpospred
#     Recall_Pos = correctpos/totalpos
#     F_measure_Pos = (2 * Precision_Pos * Recall_Pos)/(Precision_Pos + Recall_Pos)

#     Precision_Neg = correctneg/totalnegpred
#     Recall_Neg = correctneg/totalneg
#     F_measure_Neg = (2 * Precision_Neg * Recall_Neg)/(Precision_Neg + Recall_Neg)

#     print("STEP5 Accuracy:", Accuracy)
    
#     print("STEP5 Precision_Pos:", Precision_Pos)
#     print("STEP5 Recall_Pos:", Recall_Pos)
#     print("STEP5 F_measure_Pos:", F_measure_Pos)
    
#     print("STEP5 Precision_Neg:", Precision_Neg)
#     print("STEP5 Recall_Neg:", Recall_Neg)
#     print("STEP5 F_measure_Neg:", F_measure_Neg)
#     print()            
                


def testDictionarynew(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    
    negation_words = ["not", "no", "never", "without"]
    intensifiers = ["very", "extremely", "pretty", "too", "much", "more"]
    # negation_words = ["not"]
    
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score = 0
        current = 0
        for word in Words:
            if word in sentimentDictionary:
                if np.any(np.in1d(Words[current - 2 : current], negation_words)) and np.any(np.in1d(Words[current - 1], intensifiers)):
                    score+=(sentimentDictionary[word]*(-1))
                    current += 1
                elif np.any(np.in1d(Words[current - 2 : current], negation_words)):
                    score+=(sentimentDictionary[word]*(-1))
                    current += 1
                elif np.any(np.in1d(Words[current - 1 : current], intensifiers)):
                    score+=(sentimentDictionary[word]*2)
                    current += 1                    
                else:
                    score+=sentimentDictionary[word]
                    current += 1            
            else:
                current += 1

        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
    
    Accuracy = correct/total
    
    Precision_Pos = correctpos/totalpospred
    Recall_Pos = correctpos/totalpos
    F_measure_Pos = (2 * Precision_Pos * Recall_Pos)/(Precision_Pos + Recall_Pos)

    Precision_Neg = correctneg/totalnegpred
    Recall_Neg = correctneg/totalneg
    F_measure_Neg = (2 * Precision_Neg * Recall_Neg)/(Precision_Neg + Recall_Neg)

    print("STEP5 Accuracy:", Accuracy)
    
    print("STEP5 Precision_Pos:", Precision_Pos)
    print("STEP5 Recall_Pos:", Recall_Pos)
    print("STEP5 F_measure_Pos:", F_measure_Pos)
    
    print("STEP5 Precision_Neg:", Precision_Neg)
    print("STEP5 Recall_Neg:", Recall_Neg)
    print("STEP5 F_measure_Neg:", F_measure_Neg)
    print()


                
#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

# print ("Naive Bayes")



    
    
# negation_words = ['not', 'no', 'never', 'without']
# Words = ['This', 'movie', 'is', 'boring', 'and', 'I', 'think', 'it', 'is', 'not', 'movie', 'movie', 'movie', 'movie', 'good']
# Words = ['never', 'movie', 'movie', 'movie', 'movie', 'good']
# print(Words[0 : 5])
# if np.any(np.in1d(Words[0 : 5],negation_words)):
#     print("yes")

# current = 0
# score = 0
  
# for word in Words:
#     print(current)
#     if word in sentimentDictionary:
#         print(word)
#         if np.any(np.in1d(Words[current - 5 : current], negation_words)):
#             score+=sentimentDictionary[word]*(-1)
#             print(score)
#             current += 1
#         else:
#             score+=sentimentDictionary[word]
#             print(score)
#             current += 1
#     else:
#         current += 1
        

    



#run sentiment dictionary based classifier on datasets
testDictionarynew(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, -4)
testDictionarynew(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, -4)
testDictionarynew(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, -3)
            
            
            
            
            
            
            
            