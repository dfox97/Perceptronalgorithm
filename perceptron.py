#Daniel Fox
#Student ID: 201278002
#Assignment 1: COMP527
import numpy as np
import random #redundant unless random.seed/random shuffle is used
class Data(object): 
    """Main class which focuses on reading the dataset and sorting the data into     samples,features.

        filleName = name of file in string format.

        Using dictionary and arrays to store the split the data between output feature y and sample x.

        Dictionary makes it easy to select what classes to use  when it comes to classification discrimination.

        Using random to shuffle on the file data will help determine how well the algorithm performs when it is not fed with the data, 
        it has been commented out to make it easier to test the program.
    """ 
    def __init__(self,fileName):
       
        #open the file and split \n lines   
        self.fileData = open(fileName).read().splitlines()
        self.data=[]#store all final data in the list

        # randomise data set
        random.seed(2)
        random.shuffle(self.fileData)

        temp=[]#sort out y values while looping
        for i,j in enumerate(self.fileData):
            #split the data between x and y 
            split=j.split(',class-')#split the class labels [0]=x , [1]=y 

            #sample data parsed out as float instead of string
            x=np.array(split[0].split(',')).astype(float)#couldnt split data using numpy 
        
            y=split[1]
            if y not in temp:
                np.append(temp,y) 
            #append the samples and features into a data list.    
            self.data.append({'x': x, 'class-id': y})#append the dictionary
        
        #samples
        self.row = len(self.data[0]["x"]) #calculate length of each row = (4)


class Perceptron(object):
    """ Create Perceptron for Assignment

        Implimenting the perceptron is part of Question 2
        
        pos is the positive class (+1)
        neg is the negative class (-1) neg is set default at false so if user doesnt select a negative class number then it performs the 1 vs rest approach for question 3 and 4.

        maxIter= max iteration loop to train the perceptron

        D=Data class which will pass the relative information into the perceptron functions. Train data / Test data

        regularisationFactor= regularisation coefficent allows user to input regularisation coefficent for question 4. It is default set to 0 for questions 2,3,4.

        perceptronTrain = Uses the trianing data and calculate the weights
        perceptronTest= Once training is complete test the trained perceptron with the training data using the new weights calculated.
    """    

    def __init__(self, pos, neg=None,maxIter=20):
        self.pos = pos #positive class
        self.neg = neg #negative class
        self.maxIter=maxIter
    
    def perceptronTrain(self,D,regularisationFactor = 0):
        weights = np.zeros(D.row)#adding bias value and create weigths set to zero
        bias=1
        y = parseClassLabels(D,self.pos,self.neg)#call class function which returns the expected output values
        
        #loop through iterations which is at 20 for assignment
        for j in range(self.maxIter):

            correct,incorrect=0,0 #used to find testing accuracy

            #loop through the lengths of dataset
            for i in range(len(D.data)):
                x = D.data[i]["x"]#go through each x values 
                activation=np.sign(np.dot(weights,x)+bias)#activation function to determine if weights need updating

                if y[i]==0: pass #first look to ignore any outputs which are set at 0
                elif activation==y[i]:#then check if activation and expected output match
                    correct+=1 
                elif y[i]* activation <= 0:#update condition 
                    #update weights formula added (1-2*regularisationFactor) cancels out while set to 0
                    weights=(1- 2*regularisationFactor)*weights + y[i]*x
                    bias+=y[i]
                    incorrect+=1
                else:
                    incorrect+=1
        self.weights=weights
        self.accuracy=correct/(correct+incorrect)*100 #working out accuracy
        return self.accuracy #return accuracy for printing data in terminal

    def perceptronTest(self, D):
        # get labels for test dataset
        y = parseClassLabels(D,self.pos,self.neg)#get expected outputs for testing data
        correct,incorrect = 0,0
        #loop through the lengths of dataset

        for i in range(len(D.data)):
            x = D.data[i]['x']#go through each x values 
            
            activation=np.sign(np.dot(self.weights,x))#activation function to determine output

            if y[i]==0:pass #check if the expected output values are 0 then dont do anything
            elif y[i]==activation:#activation and expected output match
                correct += 1
            else:
                incorrect += 1

        self.accuracy=correct/(correct+incorrect)*100#calc accuracy
        return self.accuracy#return accuracy for printing data in terminal
    
#Used to sort class labels and allow 1vs all approach
#note didnt work while in data class
def parseClassLabels(D,pos,neg):
        #sets the class label relating to the dataset D.
        y = {}#Store the classes in dictionary
        for i in range(len(D.data)):
            classNum = D.data[i]["class-id"]
            if classNum == pos: #as user inputs a pos value this will become +1
                y[i] = 1 #key i and value 1
            elif neg: #as user inputs a neg value this will become -1
                y[i] = -1 if classNum == neg else 0
            else:y[i] = -1  #fix for 1vsall method , saved remaking a new function 
        return y

def main(): 
    """The main function runs all questions and prints accuracy to user.

    Question 2: 
        Impliment the Perceptron class.
    Question 3 compare:
        class 1 and 2
        class 2 and 3
        class 1 and 3   
    Question 4:
        Compare 1 vs all
    Question 5:
        add regularisation coefficent values to the 1 vs all appoach
        regularisation coefficent:
                [0.01, 0.1, 1.0, 10.0, 100.0]
    """
    print("-------------Question 2 and 3-------------------")
    
    train_data = Data("train.data")
    train_1 = Perceptron("1","2")
    train_2 = Perceptron("2","3")
    train_3 = Perceptron("1","3")
    
    print("Training Perceptron")
    train_1.perceptronTrain(train_data)
    train_2.perceptronTrain(train_data)
    train_3.perceptronTrain(train_data)
    train=[train_1,train_2,train_3]
    for i in train:
        print("Training Accuracy rate:%.2f%%"%i.accuracy)

    test_data = Data("test.data")
    print("\nTesting data")
    train_1.perceptronTest(test_data)
    train_2.perceptronTest(test_data)
    train_3.perceptronTest(test_data)
    for i in train:
        print("Testing Accuracy rate:%.2f%%"%i.accuracy)

    print("-----------------------------------------------")

    print("----------------Question 4---------------------")
    train_data = Data("train.data")
    
    train_1 = Perceptron("1")
    train_2 = Perceptron("2")
    train_3 = Perceptron("3")
    
    print("Training Perceptron")
    train_1.perceptronTrain(train_data)
    train_2.perceptronTrain(train_data)
    train_3.perceptronTrain(train_data)

    train=[train_1,train_2,train_3]
    for i in train:
        print("Training Accuracy rate:%.2f%%"%i.accuracy)

    test_data = Data("test.data")
    print("\nTesting data")
    train_1.perceptronTest(test_data)
    train_2.perceptronTest(test_data)
    train_3.perceptronTest(test_data)
    for i in train:
        print("Testing Accuracy rate:%.2f%%"%i.accuracy)
    print("-----------------------------------------------")

    print("--------------Question 5-----------------------")

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

    print("-----------------------------------------------")

if __name__ == '__main__':
    main()