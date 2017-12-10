import numpy as np
import pickle
import sys
'''
Websites and resources I referred for understanding of full batch gradient descent:
https://www.analyticsvidhya.com/
Andrew Ng's Machine Learning course on coursera
People whose words/tips I am grateful for: Aishwarya Dhage, Pulkit Maloo, Vaibhav Shah
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

def accuracy_converter(y):
	a=[]
	for i in y:
		a=np.append(a,np.asmatrix(np.argmax(i)))
	a=np.asmatrix(a)
	return a

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
		alpha=0.0001
		input_neurons=192
		hidden_neurons=5
		output_neurons=4
		weight_input_hidden=np.asmatrix(np.random.rand(input_neurons,hidden_neurons))-np.asmatrix(np.random.rand(input_neurons,hidden_neurons))
		bias_input_hidden=np.asmatrix(np.random.rand(1,hidden_neurons))-np.asmatrix(np.random.rand(1,hidden_neurons))
		weight_hidden_output=np.asmatrix(np.random.rand(hidden_neurons,output_neurons))-np.asmatrix(np.random.rand(hidden_neurons,output_neurons))
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
		pickle.dump([weight_input_hidden,weight_hidden_output,bias_input_hidden,bias_hidden_output],open("weights_data.p", "wb"))
	else:
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
		x=np.sum(test_converted_output==a_output)
		print "Test Accuracy:",float(x)*100/number_of_test,"%"
		a_output=back_converter(a_output,number_of_test)
		test_names=np.transpose(np.asmatrix(test_names))
		tofile=np.hstack((test_names,a_output))
		tofile.astype('|S10')
		np.savetxt('output.txt', tofile, delimiter=' ',fmt="%s")

