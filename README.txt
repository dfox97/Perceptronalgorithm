COMP527 Assignment 1 - Created by Daniel Fox 

Requirements for this program are:
Numpy 
Random

When this file run through the terminal all data should print out in order of question in attempt to make it easy and simple for the marker. 
The expected results can be shown at the end of the README file.

The program has random shuffle impliemented in the source code it can be commented if needed on line 25/26.


Feel free to comment out the tests if needed. 
Below the results show how to individually run the code for each question.
Individual commands to run each question:
*******************************************
#question 3 example

   train_data = Data("train.data")
   train_1 = Perceptron("1","2")
   train_1.perceptronTrain(train_data)
   print("Testing Accuracy rate:%.2f%%"%train_1.accuracy)
    
   test_data = Data("test.data")
   train_1.perceptronTest(test_data)
   print("Training Accuracy rate:%.2f%%"%train_1.accuracy)
*******************************************
#question 4 example
    train_data = Data("train.data")
    train_2 = Perceptron("2")
    train_2.perceptronTrain(train_data)
    print("Testing Accuracy rate:%.2f%%"%train_2.accuracy)

    test_data = Data("test.data")
    train_2.perceptronTest(test_data)
    print("Training Accuracy rate:%.2f%%"%train_2.accuracy)

*****************************************
#question 5 example
   train_data = Data("train.data")
   train_3 = Perceptron("2")
   train_3.perceptronTrain(train_data,0.01)
   print("Testing Accuracy rate:%.2f%%"%train_3.accuracy)
    
   test_data = Data("test.data")
   train_3.perceptronTest(test_data)
   print("Training Accuracy rate:%.2f%%"%train_3.accuracy)

*****************************************
#ANOTHER WAY FOR QUESTION 5 to print ALL testing
#If you want to run all at once an test each class with each regularisation facotor

    train_data = Data("train.data")
    regularisation = [0.01, 0.1, 1.0, 10.0, 100.0]
    train_1 = Perceptron("1")
    train_2 = Perceptron("2")
    train_3 = Perceptron("3")


    test_data = Data("test.data")
    print("Testing data")
    for i in (regularisation):
        print("\nRegularisation factor:%.2f\n"%i)
        train_1.perceptronTrain(train_data,i)
        #print("Training Accuracy rate:%.2f%%"%train_1.accuracy)#testing the training accuracy
        train_1.perceptronTest(test_data)
        print("Testing Accuracy rate:%.2f%%"%train_1.accuracy)
        
        train_2.perceptronTrain(train_data,i)
        #print("Training Accuracy rate:%.2f%%"%train_2.accuracy)#testing the training accuracy
        train_2.perceptronTest(test_data)
        print("Testing Accuracy rate:%.2f%%"%train_2.accuracy)

        train_3.perceptronTrain(train_data,i)
        #print("Training Accuracy rate:%.2f%%"%train_3.accuracy)#testing the training accuracy
        train_3.perceptronTest(test_data)
        print("Testing Accuracy rate:%.2f%%"%train_3.accuracy)

*****************************************
>>>>>>>>>>>EXPECTED RESULTS ON RUN<<<<<<<<<<<<<<<<<<<<
-------------Question 2 and 3-------------------
Training Perceptron
Training Accuracy rate:100.00%
Training Accuracy rate:83.75%
Training Accuracy rate:100.00%

Testing data
Testing Accuracy rate:100.00%
Testing Accuracy rate:95.00%
Testing Accuracy rate:100.00%
-----------------------------------------------
----------------Question 4---------------------
Training Perceptron
Training Accuracy rate:100.00%
Training Accuracy rate:65.83%
Training Accuracy rate:89.17%

Testing data
Testing Accuracy rate:100.00%
Testing Accuracy rate:66.67%
Testing Accuracy rate:96.67%
-----------------------------------------------
--------------Question 5-----------------------
Testing data

Regularisation factor:0.01

Testing Accuracy rate:100.00%
Testing Accuracy rate:66.67%
Testing Accuracy rate:66.67%

Regularisation factor:0.10

Testing Accuracy rate:100.00%
Testing Accuracy rate:60.00%
Testing Accuracy rate:33.33%

Regularisation factor:1.00

Testing Accuracy rate:66.67%
Testing Accuracy rate:66.67%
Testing Accuracy rate:33.33%

Regularisation factor:10.00

Testing Accuracy rate:66.67%
Testing Accuracy rate:66.67%
Testing Accuracy rate:33.33%

Regularisation factor:100.00

Testing Accuracy rate:66.67%
Testing Accuracy rate:66.67%
Testing Accuracy rate:33.33%
-----------------------------------------------





