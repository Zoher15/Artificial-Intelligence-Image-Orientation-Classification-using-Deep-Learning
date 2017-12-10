#!/usr/bin/env python
import numpy as np
import pickle
import sys
'''
Authors: Zoher Kachwala
Websites and resources I referred for understanding of full batch gradient descent:
https://www.analyticsvidhya.com/
Andrew Ng's Machine Learning course on coursera
People whose words/tips I am grateful for: Aishwarya Dhage, Pulkit Maloo, Vaibhav Shah
neurons(hidden layer)		alpha		train_accuracy		test_accuracy
	4			0.0001		72%			70%
	4			0.001		68%			65%
	5			0.0001		72%			70%
	6			0.001		70%			67%
	8			0.001		70%			68%
	64			all		30%(overflow)		30%(overflow)
	192			all		25%(oveflow)		25%(overflow)

I was initially trying to build stochastic gradient but it was too slow and testing could not be as fast as I wanted. Plus I had a huge bug 
where I was not initializing the weights from -0.5 to 0.5. For a whole week I was getting 25% accuracy.

Then I decided to look methods to implemement full batch gradient descent. Found it to be better. But still no luck with accuracy.Stuck at 25%
Tried Normalization. Tried shuffling of training data. Tried all kinds of complicated things. Nothing worked. (Why would it, this was simpler than I thought)

Then after finally giving up I tried to go back to basics and I tried implement XOR using the same code. AND IT WORKED!
First positive sign after two weeks, HURRAY! I implemented the mighty XOR after two weeks

So I persisted with different alphas and different neurons. Still stuck at 25% (random guess basically).
Finally due to useful discussions with Aishwarya, I changed my weight initialization and my net started throwing results above 25%
I started with 4 neurons which was giving me great accuracy btw: 70% on train!
But while testing (due to random initialization) 4 neurons would occasionally produce only 40% accuracy. Rare but worrying
So I increased it to 5 neurons, much more stable even with rigorous testing.
Tried increasing neurons but did not improve my accuracy. So stuck with 5.
'''
def expected_output_converter(y):
	outputdict={0:np.asmatrix([1,0,0,0]),90:np.asmatrix([0,1,0,0]),180:np.asmatrix([0,0,1,0]),270:np.asmatrix([0,0,0,1])}
	a=outputdict[y[0]]
	for i in y[1:]:
		a=np.vstack((a,outputdict[i]))
	return a

#activation function
def sigmoid(u):
	return 1/(1+np.exp(-u))
#derivative of activation function
def dsigmoid(a):
    return np.multiply(a,(1.0 - a))

def back_converter(u,length):
	outputdict={0:np.asmatrix(0),1:np.asmatrix(90),2:np.asmatrix(180),3:np.asmatrix(270)}
	a=outputdict[u[0,0]]
	for i in range(1,length):
		a=np.vstack((a,outputdict[u[0,i]]))
	return a
#uses argmax to calculate the answer of the output
def accuracy_converter(y):
	a=[]
	for i in y:
		a=np.append(a,np.asmatrix(np.argmax(i)))
	a=np.asmatrix(a)
	return a
#neural net starts here
def neural_net(parameter, t_fname, model_fname):
	########
	if parameter=='train':
		#taking Values for TRAIN
		train=np.genfromtxt(t_fname,dtype='str')
		train_names=train[:,0]
		train_num=number_of_train=len(train_names)
		train_expected_outputx=train[:,1].astype(np.float)
		train_expected_output=expected_output_converter(train_expected_outputx)
		train_data=(train[:,2:].astype(np.float))
		train_converted_output=accuracy_converter(train_expected_output)
		#setting parameters
		epochs=1000
		#magic number
		alpha=0.0001
		input_neurons=192
		#sign of simplicity: 5 neurons
		hidden_neurons=5
		output_neurons=4
		#random initialization of weight
		weight_input_hidden=np.asmatrix(np.random.rand(input_neurons,hidden_neurons))-np.asmatrix(np.random.rand(input_neurons,hidden_neurons))
		#random initializaion of bias
		bias_input_hidden=np.asmatrix(np.random.rand(1,hidden_neurons))-np.asmatrix(np.random.rand(1,hidden_neurons))
		#random initialization of weight
		weight_hidden_output=np.asmatrix(np.random.rand(hidden_neurons,output_neurons))-np.asmatrix(np.random.rand(hidden_neurons,output_neurons))
		#random initializaion of bias
		bias_hidden_output=np.asmatrix(np.random.rand(1,output_neurons))-np.asmatrix(np.random.rand(1,output_neurons))
		for i in range(epochs):
			#FORWARD MARCH!
			a_hidden=sigmoid(np.dot(train_data,weight_input_hidden)+bias_input_hidden)
			a_output=sigmoid(np.dot(a_hidden,weight_hidden_output)+bias_hidden_output)
			#BACKWARD SWIM!
			error_difference=train_expected_output-a_output
			#print i,":Error:",0.5*np.sum(np.square(error_difference))
			delta_output=np.multiply(error_difference,dsigmoid(a_output))
			delta_hidden=np.multiply(np.dot(delta_output,np.transpose(weight_hidden_output)),dsigmoid(a_hidden))
			#Hidden-Output Updates
			weight_hidden_output+=alpha*np.dot(np.transpose(a_hidden),delta_output)
			bias_hidden_output+=alpha*np.sum(delta_output)
			#Input-Hidden Updates
			weight_input_hidden+=alpha*np.dot(np.transpose(train_data),delta_hidden)
			bias_input_hidden+=alpha*np.sum(delta_hidden)
		#pickling trained weights
		pickle.dump([weight_input_hidden,weight_hidden_output,bias_input_hidden,bias_hidden_output],open("weights_data.p", "wb"))
	else:
		#unpickling saved weights
		yoarray=pickle.load(open("weights_data.p","rb"))
		weight_input_hidden=yoarray[0]
		weight_hidden_output=yoarray[1]
		bias_input_hidden=yoarray[2]
		bias_hidden_output=yoarray[3]
		#Taking values FOR TEST
		test=np.genfromtxt(t_fname,dtype='str')
		test_names=test[:,0]
		number_of_test=len(test_names)
		test_expected_outputx=test[:,1].astype(np.float)
		test_expected_output=expected_output_converter(test_expected_outputx)
		test_data=(test[:,2:].astype(np.float))
		test_converted_output=accuracy_converter(test_expected_output)
		####TEST DATA
		a_hidden=sigmoid(np.dot(test_data,weight_input_hidden)+bias_input_hidden)
		a_output=sigmoid(np.dot(a_hidden,weight_hidden_output)+bias_hidden_output)
		a_output=accuracy_converter(a_output)
		#counting accuracy in term x
		x=np.sum(test_converted_output==a_output)
		print ("Test Accuracy:",float(x)*100/number_of_test,"%")
		a_output=back_converter(a_output,number_of_test)
		test_names=np.transpose(np.asmatrix(test_names))
		tofile=np.hstack((test_names,a_output))
		tofile.astype('|S10')
		#writing to output file
		np.savetxt('output.txt', tofile, delimiter=' ',fmt="%s")

