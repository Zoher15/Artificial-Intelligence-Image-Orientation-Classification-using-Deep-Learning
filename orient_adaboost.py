#!/usr/bin/env python


# Training the file might take sometime depending on the size of the training file. However, testing doesn't take too much time.

# The below code implements adaboost algorithm. 

# I've used a simple decision stump. Randomly picked two pixel values from every image
# of the training dataset and checked which one is greater. I've done this for all the images and taken a vote. 
# The same thing is done during testing.

# I've used 10 classifiers to predict the orientation of an image.

# The code contains two important functons, 1 for training (decision_stumps()) and the other for testing (test_adaboost() ).

# I have implemented the "one vs one" model. I have six models, 0-90, 0-180,0-270,90-180 and so on for every class. 
# During testing, a vote of all these models is taken and the orientation with the highest vote is assigned to the testing image.

# At first, I have classified the data into six sub-datasets. First dataset contains only 0-90 orientation images, the second 0-180
# and so on.

# These datasets are then used to compare the pixel values and get a vote.

# Later, I write, 1. the features used, 2. alpha values and 3. the predcited values of all the 6 models into the mode_file.txt file.

# This file is then read into and passed to the testing function.

# The test_adaboost function takes input as the model file generated during training and the test file passed as a parameter from terminal.

# The function then compares the same two features (Pixel values) used during training and checks which feature is greater.

# Assigns an orientation accordingly. 

# This is done for each model and then a vote is taken across 6 models. The highest vote is the predicted orientation.

# Unfortunately, even after 5 long and agonizing days of debugging and multiple attempts of rewriting the code our accuracy still stands
# at a number close to 25%. 


import numpy as np
import math
import random


def read_train_list(t_fname):

	photo_id=list()
	expected_output=list()
	data=list()
	all_data=list()

	with open(t_fname,"r") as file:
		for i in file:
			lines = i.split()
			photo_id.append(str(lines[0]))
			expected_output.append(int(lines[1]))
			data.append(map(int,lines[2:]))
			all_data.append(lines)

	# weights=np.zeros(len(photo_id))
	# for i in range(len(photo_id)):
	# 	weights[i]=float(1)/len(photo_id)

	return photo_id,expected_output,data, all_data

# read_train_list("test-data.txt")


def decision_stumps(filename,model_fname):
	x=read_train_list(filename)
	photo_id=x[0]
	expected_output=x[1]
	data=x[2]
	# weights=x[3]
	all_data=x[3]
	no_of_images=len(photo_id)
	data_0_90=list()
	data_0_180=list()
	data_0_270=list()
	data_90_180=list()
	data_90_270=list()
	data_180_270=list()
	feature1=list()
	feature2=list()
	

	for i in range(no_of_images):
		if(expected_output[i]==0 or expected_output[i]==90):
			data_0_90.append(map(int,all_data[i][1:]))

		if(expected_output[i]==0 or expected_output[i]==180):
			data_0_180.append(map(int,all_data[i][1:]))

		if(expected_output[i]==0 or expected_output[i]==270):
			data_0_270.append(map(int,all_data[i][1:]))

		if(expected_output[i]==90 or expected_output[i]==180):
			data_90_180.append(map(int,all_data[i][1:]))

		if(expected_output[i]==90 or expected_output[i]==270):
			data_90_270.append(map(int,all_data[i][1:]))

		if(expected_output[i]==180 or expected_output[i]==270):
			data_180_270.append(map(int,all_data[i][1:]))

	

	# print len(data_0_90)

	len_0_90 = len(data_0_90)
	positive_count_0_90=[0 for i in range(len_0_90)]
	positive_count_0_180=[0 for i in range(len(data_0_180))]
	positive_count_0_270=[0  for i in range(len(data_0_270))]
	positive_count_90_180=[0 for i in range(len(data_90_180))]
	positive_count_90_270=[0 for i in range(len(data_90_270))]
	positive_count_180_270=[0 for i in range(len(data_180_270))]
	positive_vote_90=0
	positive_vote_0=0
	positive_vote_180=0
	positive_vote_270=0
	predicted_label_0_90=[0 for i in range(len_0_90)]
	negative_vote_0=0
	negative_vote_90=0
	negative_count_0_90=[0 for i in range(len_0_90)]
	negative_count_0_180=[0 for i in range(len(data_0_180))]
	negative_vote_180=0
	negative_vote_270=0
	negative_count_0_270=[0 for i in range(len(data_0_270))]
	negative_count_90_180=[0 for i in range(len(data_90_180))]
	negative_count_90_270=[0 for i in range(len(data_90_270))]
	negative_count_180_270=[0 for i in range(len(data_180_270))]
	predicted_label_0_180=[0 for i in range(len(data_0_180))]
	predicted_label_0_270=[0 for i in range(len(data_0_270))]
	predicted_label_90_180=[0 for i in range(len(data_90_180))]
	predicted_label_90_270=[0 for i in range(len(data_90_270))]
	predicted_label_180_270=[0 for i in range(len(data_180_270))]
	right=[0 for i in range(len(data_0_90))]
	right_0_180=[0 for i in range(len(data_0_180))]
	right_0_270=[0 for i in range(len(data_0_270))]
	right_90_180=[0 for i in range(len(data_90_180))]
	right_90_270=[0 for i in range(len(data_90_270))]
	right_180_270=[0 for i in range(len(data_180_270))]
	weights_0_90=[float(1)/len_0_90 for i in range(len_0_90)]
	weights_0_180=[float(1)/len(data_0_180) for i in range(len(data_0_180))]
	weights_0_270=[float(1)/len(data_0_270) for i in range(len(data_0_270))]
	weights_90_180=[float(1)/len(data_90_180) for i in range(len(data_90_180))]
	weights_90_270=[float(1)/len(data_90_270) for i in range(len(data_90_270))]
	weights_180_270=[float(1)/len(data_180_270) for i in range(len(data_180_270))]
	alpha_0_90=list()
	alpha_0_180=list()
	alpha_0_270=list()
	alpha_90_180=list()
	alpha_90_270=list()
	alpha_180_270=list()

	blue_feature = []
	for i in range(2,191,3):
		blue_feature.append(i)


	for k in range(10):
		feature1.append(random.randint(0,191))
		feature2.append(random.randint(0,191))
		# feature1.append(random.choice(blue_feature))
		# feature2.append(random.choice(blue_feature))

		error_0_90=0
		error_0_180=0
		error_0_270=0
		error_90_180=0
		error_90_270=0
		error_180_270=0


		for i in range(len_0_90):
			if(data_0_90[i][feature1[k]] >= data_0_90[i][feature2[k]]):
				if(data_0_90[i][0]==0):
					positive_count_0_90[i]=1
					positive_vote_0+=1
				if(data_0_90[i][0]==90):
					positive_count_0_90[i]=-1
					positive_vote_90+=1

				# if(positive_count_0_90[i]==1 or positive_count_0_90[i]==-1):
				if(positive_vote_0>positive_vote_90):
					predicted_label_0_90[i]=0
				else:
					predicted_label_0_90[i]=90

			if(data_0_90[i][feature1[k]] < data_0_90[i][feature2[k]] ):
				if(data_0_90[i][0]==0):
					negative_count_0_90[i]=1
					negative_vote_0+=1
				if(data_0_90[i][0]==90):
					negative_count_0_90[i]=-1
					negative_vote_90+=1

				if(negative_vote_0> negative_vote_90):
					predicted_label_0_90[i]=0
				else:
					predicted_label_0_90[i]=90

			if(data_0_90[i][0]==predicted_label_0_90[i]):
				right[i]=1
			else:
				right[i]=-1

			accuracy=float(right.count(1))/len_0_90


		# for i in range(len_0_90):
			if(right[i]==-1):
				error_0_90 = error_0_90 + weights_0_90[i]

			if(right[i]==1):
				weights_0_90[i]=weights_0_90[i]*(error_0_90/(1-error_0_90))

			weights_0_90[i]=weights_0_90[i]/sum(weights_0_90)
	
		alpha_0_90.append(math.log((1-error_0_90)/error_0_90))


		positive_vote_0=0
		negative_vote_0=0

		for i in range(len(data_0_180)):
			if(data_0_180[i][feature1[k]] >= data_0_180[i][feature2[k]]):
				if(data_0_180[i][0]==0):
					positive_count_0_180[i]=1
					positive_vote_0+=1
				if(data_0_180[i][0]==180):
					positive_count_0_180[i]=-1
					positive_vote_180+=1

				# if(positive_count_0_90[i]==1 or positive_count_0_90[i]==-1):
				if(positive_vote_0>positive_vote_180):
					predicted_label_0_180[i]=0
				else:
					predicted_label_0_180[i]=180


			if(data_0_180[i][feature1[k]] < data_0_180[i][feature2[k]] ):
				if(data_0_180[i][0]==0):
					negative_count_0_180[i]=1
					negative_vote_0+=1
				if(data_0_180[i][0]==180):
					negative_count_0_180[i]=-1
					negative_vote_180+=1

				if(negative_vote_0> negative_vote_180):
					predicted_label_0_180[i]=0
				else:
					predicted_label_0_180[i]=180


			if(data_0_180[i][0]==predicted_label_0_180[i]):
				right_0_180[i]=1
			else:
				right_0_180[i]=-1

			accuracy=float(right_0_180.count(1))/len(data_0_180)


		# for i in range(len_0_90):
			if(right_0_180[i]==-1):
				error_0_180 = error_0_180 + weights_0_180[i]

			if(right_0_180[i]==1):
				weights_0_180[i]=weights_0_180[i]*(error_0_180/(1-error_0_180))

			weights_0_180[i]=weights_0_180[i]/sum(weights_0_180)
	
		alpha_0_180.append(math.log((1-error_0_180)/error_0_180))

		
		positive_vote_0=0
		negative_vote_0=0

		for i in range(len(data_0_270)):
			if(data_0_270[i][feature1[k]] >= data_0_270[i][feature2[k]]):
				if(data_0_270[i][0]==0):
					positive_count_0_270[i]=1
					positive_vote_0+=1
				if(data_0_270[i][0]==270):
					positive_count_0_270[i]=-1
					positive_vote_270+=1

				# if(positive_count_0_90[i]==1 or positive_count_0_90[i]==-1):
				if(positive_vote_0>positive_vote_270):
					predicted_label_0_270[i]=0
				else:
					predicted_label_0_270[i]=270


			if(data_0_270[i][feature1[k]] < data_0_270[i][feature2[k]] ):
				if(data_0_270[i][0]==0):
					negative_count_0_270[i]=1
					negative_vote_0+=1
				if(data_0_270[i][0]==270):
					negative_count_0_270[i]=-1
					negative_vote_270+=1

				if(negative_vote_0> negative_vote_270):
					predicted_label_0_270[i]=0
				else:
					predicted_label_0_270[i]=270


			if(data_0_270[i][0]==predicted_label_0_270[i]):
				right_0_270[i]=1
			else:
				right_0_270[i]=-1

			accuracy=float(right_0_270.count(1))/len(data_0_270)


			# for i in range(len_0_90):
			if(right_0_270[i]==-1):
				error_0_270 = error_0_270 + weights_0_270[i]

			if(right_0_270[i]==1):
				weights_0_270[i]=weights_0_270[i]*(error_0_270/(1-error_0_270))

			weights_0_270[i]=weights_0_270[i]/sum(weights_0_270)
		
		alpha_0_270.append(math.log((1-error_0_270)/error_0_270))


		positive_vote_90=0
		negative_vote_90=0
		positive_vote_180=0
		negative_vote_180=0

		for i in range(len(data_90_180)):
			if(data_90_180[i][feature1[k]] >= data_90_180[i][feature2[k]]):
				if(data_90_180[i][0]==90):
					positive_count_90_180[i]=1
					positive_vote_90+=1
				if(data_90_180[i][0]==180):
					positive_count_90_180[i]=-1
					positive_vote_180+=1

				# if(positive_count_0_90[i]==1 or positive_count_0_90[i]==-1):
				if(positive_vote_90>positive_vote_180):
					predicted_label_90_180[i]=90
				else:
					predicted_label_90_180[i]=180


			if(data_90_180[i][feature1[k]] < data_90_180[i][feature2[k]] ):
				if(data_90_180[i][0]==90):
					negative_count_90_180[i]=1
					negative_vote_90+=1
				if(data_90_180[i][0]==180):
					negative_count_90_180[i]=-1
					negative_vote_180+=1

				if(negative_vote_90> negative_vote_180):
					predicted_label_90_180[i]=90
				else:
					predicted_label_90_180[i]=180


			if(data_90_180[i][0]==predicted_label_90_180[i]):
				right_90_180[i]=1
			else:
				right_90_180[i]=-1

			accuracy=float(right_90_180.count(1))/len(data_90_180)


			# for i in range(len_0_90):
			if(right_90_180[i]==-1):
				error_90_180 = error_90_180 + weights_90_180[i]

			if(right_90_180[i]==1):
				weights_90_180[i]=weights_90_180[i]*(error_90_180/(1-error_90_180))

			weights_90_180[i]=weights_90_180[i]/sum(weights_90_180)
		
		alpha_90_180.append(math.log((1-error_90_180)/error_90_180))


		positive_vote_90=0
		negative_vote_90=0
		positive_vote_270=0
		negative_vote_270=0

		for i in range(len(data_90_270)):
			if(data_90_270[i][feature1[k]] >= data_90_270[i][feature2[k]]):
				if(data_90_270[i][0]==90):
					positive_count_90_270[i]=1
					positive_vote_90+=1
				if(data_90_270[i][0]==270):
					positive_count_90_270[i]=-1
					positive_vote_270+=1

				# if(positive_count_0_90[i]==1 or positive_count_0_90[i]==-1):
				if(positive_vote_90>positive_vote_270):
					predicted_label_90_270[i]=90
				else:
					predicted_label_90_270[i]=270


			if(data_90_270[i][feature1[k]] < data_90_270[i][feature2[k]] ):
				if(data_90_270[i][0]==90):
					negative_count_90_270[i]=1
					negative_vote_90+=1
				if(data_90_270[i][0]==270):
					negative_count_90_270[i]=-1
					negative_vote_270+=1

				if(negative_vote_90> negative_vote_270):
					predicted_label_90_270[i]=90
				else:
					predicted_label_90_270[i]=270


			if(data_90_270[i][0]==predicted_label_90_270[i]):
				right_90_270[i]=1
			else:
				right_90_270[i]=-1

			accuracy=float(right_90_270.count(1))/len(data_90_270)


			# for i in range(len_0_90):
			if(right_90_270[i]==-1):
				error_90_270 = error_90_270 + weights_90_270[i]

			if(right_90_270[i]==1):
				weights_90_270[i]=weights_90_270[i]*(error_90_270/(1-error_90_270))

			weights_90_270[i]=weights_90_270[i]/sum(weights_90_270)
		
		alpha_90_270.append(math.log((1-error_90_270)/error_90_270))



		positive_vote_180=0
		negative_vote_180=0
		positive_vote_270=0
		negative_vote_270=0

		for i in range(len(data_180_270)):
			if(data_180_270[i][feature1[k]] >= data_180_270[i][feature2[k]]):
				if(data_180_270[i][0]==180):
					positive_count_180_270[i]=1
					positive_vote_180+=1
				if(data_180_270[i][0]==270):
					positive_count_180_270[i]=-1
					positive_vote_270+=1

				# if(positive_count_0_90[i]==1 or positive_count_0_90[i]==-1):
				if(positive_vote_180>positive_vote_270):
					predicted_label_180_270[i]=180
				else:
					predicted_label_180_270[i]=270

			if(data_180_270[i][feature1[k]] < data_180_270[i][feature2[k]] ):
				if(data_180_270[i][0]==180):
					negative_count_180_270[i]=1
					negative_vote_180+=1
				if(data_180_270[i][0]==270):
					negative_count_180_270[i]=-1
					negative_vote_270+=1

				if(negative_vote_180> negative_vote_270):
					predicted_label_180_270[i]=180
				else:
					predicted_label_180_270[i]=270


			if(data_180_270[i][0]==predicted_label_180_270[i]):
				right_180_270[i]=1
			else:
				right_180_270[i]=-1

			accuracy=float(right_180_270.count(1))/len(data_180_270)


			# for i in range(len_0_90):
			if(right_180_270[i]==-1):
				error_180_270 = error_180_270 + weights_180_270[i]

			if(right_180_270[i]==1):
				weights_180_270[i]=weights_180_270[i]*(error_180_270/(1-error_180_270))

			weights_180_270[i]=weights_180_270[i]/sum(weights_180_270)
		
		alpha_180_270.append(math.log((1-error_180_270)/error_180_270))

	with open(model_fname,"w") as f:
		f.write(" ".join(str(i) for i in feature1) + "\n")
		f.write(" ".join(str(i) for i in feature2) + "\n" )
		f.write(" ".join(str(i) for i in weights_0_90) +"\n" )
		f.write(" ".join(str(i) for i in alpha_0_90) + "\n" )
		f.write(" ".join(str(i) for i in predicted_label_0_90)+"\n" )

		f.write(" ".join(str(i) for i in weights_0_180) +"\n" )
		f.write(" ".join(str(i) for i in alpha_0_180) + "\n" )
		f.write(" ".join(str(i) for i in predicted_label_0_180)+"\n" )

		f.write(" ".join(str(i) for i in weights_0_270) +"\n" )
		f.write(" ".join(str(i) for i in alpha_0_270) + "\n" )
		f.write(" ".join(str(i) for i in predicted_label_0_270)+"\n" )

		f.write(" ".join(str(i) for i in weights_90_180) +"\n" )
		f.write(" ".join(str(i) for i in alpha_90_180) + "\n" )
		f.write(" ".join(str(i) for i in predicted_label_90_180)+"\n" )

		f.write(" ".join(str(i) for i in weights_90_270) +"\n" )
		f.write(" ".join(str(i) for i in alpha_90_270) + "\n" )
		f.write(" ".join(str(i) for i in predicted_label_90_270)+"\n" )

		f.write(" ".join(str(i) for i in weights_180_270) +"\n" )
		f.write(" ".join(str(i) for i in alpha_180_270) + "\n" )
		f.write(" ".join(str(i) for i in predicted_label_180_270)+"\n" )



	# print feature1
	# print feature2




		# print right_180_270.count(1)
		# print predicted_label_180_270
		# print predicted_label_0_180,accuracy
		# print negative_vote_0,negative_vote_90, predicted_label_0_90,accuracy
		# print weights_0_90
	# return feature1,feature2,weights_0_90,alpha_0_90,predicted_label_0_90,weights_0_180,alpha_0_180,predicted_label_0_180,weights_0_270,alpha_0_270,predicted_label_0_270,weights_90_180,alpha_90_180,predicted_label_90_180,weights_90_270,alpha_90_270,predicted_label_90_270,weights_180_270,alpha_180_270,predicted_label_180_270

def test_adaboost(t_fname,model_fname):
	x=read_model_file(model_fname)
	y=read_train_list(t_fname)
	photo_id=y[0]
	no_of_images_test = len(y[0])
	expected_output=y[1]
	test_data=y[2]
	feature1=x[0]
	feature2=x[1]
	weights_0_90=x[2]
	alpha_0_90=x[3]
	predicted_label_0_90=x[4]

	weights_0_180=x[5]
	alpha_0_180=x[6]
	predicted_label_0_180=x[7]

	weights_0_270=x[8]
	alpha_0_270=x[9]
	predicted_label_0_270=x[10]

	weights_90_180=x[11]
	alpha_90_180=x[12]
	predicted_label_90_180=x[13]

	weights_90_270=x[14]
	alpha_90_270=x[15]
	predicted_label_90_270=x[16]

	weights_180_270=x[17]
	alpha_180_270=x[18]
	predicted_label_180_270=x[19]

	predict_count_0_90=list()
	predict_count_0_180=list()
	predict_count_0_270=list()
	predict_count_90_180=list()
	predict_90_270=list()
	predict_count_180_270=list()
	predict_count_90_270=list()
	predict_count_180_270=list()
	predict_0_90=list()
	predict_0_180=list()
	predict_0_270=list()
	predict_90_180=list()
	predict_90_270=list()
	predict_180_270=list()

	accuracy_count=0
	for i in range(no_of_images_test):
		for j in range(len(feature1)):
			if(test_data[i][feature1[j]] > test_data[i][feature2[j]] ):
				if(predicted_label_0_90[i]==0):
					predict_count_0_90.append((1,alpha_0_90[j]))
				if(predicted_label_0_90[i]==90):
					predict_count_0_90.append((-1,alpha_0_90[j]))

		answer=0
		for j in predict_count_0_90:
			answer+= j[0] * j[1]

		if(answer>0):
			predict_0_90.append(0)
		else:
			predict_0_90.append(90)

	for i in range(no_of_images_test):
		for j in range(len(feature1)):
			if(test_data[i][feature1[j]]>test_data[i][feature2[j]]):
				if(predicted_label_0_180[i]==0):
					predict_count_0_180.append((1,alpha_0_180[j]))
				if(predicted_label_0_180[i]==180):
					predict_count_0_180.append((-1,alpha_0_180[j]))

		answer_0_180=0
		for j in predict_count_0_180:
			answer_0_180+=j[0]*j[1]

		if(answer_0_180>0):
			predict_0_180.append(0)
		else:
			predict_0_180.append(180)


	for i in range(no_of_images_test):
		for j in range(len(feature1)):
			if(test_data[i][feature1[j]]>test_data[i][feature2[j]]):
				if(predicted_label_0_270[i]==0):
					predict_count_0_270.append((1,alpha_0_270[j]))
				if(predicted_label_0_270[i]==270):
					predict_count_0_270.append((-1,alpha_0_270[j]))

		answer_0_270=0
		for j in predict_count_0_270:
			answer_0_270+=j[0]*j[1]

		if(answer_0_270>0):
			predict_0_270.append(0)
		else:
			predict_0_270.append(270)


	for i in range(no_of_images_test):
		for j in range(len(feature1)):
			if(test_data[i][feature1[j]]>test_data[i][feature2[j]]):
				if(predicted_label_90_180[i]==90):
					predict_count_90_180.append((1,alpha_90_180[j]))
				if(predicted_label_90_180[i]==180):
					predict_count_90_180.append((-1,alpha_90_180[j]))

		answer_90_180=0
		for j in predict_count_90_180:
			answer_90_180+=j[0]*j[1]

		if(answer_90_180>0):
			predict_90_180.append(90)
		else:
			predict_90_180.append(180)


	for i in range(no_of_images_test):
		for j in range(len(feature1)):
			if(test_data[i][feature1[j]]>test_data[i][feature2[j]]):
				if(predicted_label_90_270[i]==90):
					predict_count_90_270.append((1,alpha_90_270[j]))
				if(predicted_label_90_270[i]==270):
					predict_count_90_270.append((-1,alpha_90_270[j]))

		answer_90_270=0
		for j in predict_count_90_270:
			answer_90_270+=j[0]*j[1]

		if(answer_90_270>0):
			predict_90_270.append(90)
		else:
			predict_90_270.append(270)



	for i in range(no_of_images_test):
		for j in range(len(feature1)):
			if(test_data[i][feature1[j]]>test_data[i][feature2[j]]):
				if(predicted_label_180_270[i]==180):
					predict_count_180_270.append((1,alpha_180_270[j]))
				if(predicted_label_180_270[i]==270):
					predict_count_180_270.append((-1,alpha_180_270[j]))

		answer_180_270=0
		for j in predict_count_180_270:
			answer_180_270+=j[0]*j[1]

		if(answer_180_270>0):
			predict_180_270.append(180)
		else:
			predict_180_270.append(270)

	
	final_count=list()
	final_predict=list()
	for i in range(no_of_images_test):
		final_count.append(predict_0_90[i])
		final_count.append(predict_0_180[i])
		final_count.append(predict_0_270[i])
		final_count.append(predict_90_180[i])
		final_count.append(predict_90_270[i])
		final_count.append(predict_180_270[i])
		final_predict.append(max(final_count,key=final_count.count))
		final_count=[]

	# print final_predict

	for j in range(len(test_data)):
		# if(expected_output[i]==0 or expected_output[i]==90):
		if(expected_output[j]==final_predict[j]):
			accuracy_count+=1

	accuracy=float(accuracy_count)/len(test_data)

	print (accuracy)

	with open("output.txt","w") as f:
		# f.write(" ".join(photo_id))
		# f.write(" ".join(map(str,final_predict))+"\n")
		for i,j in zip(photo_id,final_predict):
			f.write(('{0} {1}\n'.format(i,j)))


	# print final_count_0,final_count_90,final_count_180,final_count_270
	# print final_count_0
	# print predict_0_90 , predict_0_180,predict_0_270,predict_90_180,predict_90_270,predict_180_270

	# print predict_90_180, predict_90_270, predict_180_270
	# print type(expected_output[0])



def read_model_file(t_fname):

	lines=list()
	with open(t_fname,"r") as file:
		for each_line in file:
			#x  = i.split("\n")
			xlist = each_line.split(" ")
			lines.append([float(val) for val in xlist])

	#print lines[0]


	feature1=map(int,lines[0])
	feature2=map(int,lines[1])
	weights_0_90=lines[2]
	alpha_0_90=lines[3]
	predicted_label_0_90=lines[4]
	weights_0_180=lines[5]
	alpha_0_180=lines[6]
	predicted_label_0_180=lines[7]
	weights_0_270=lines[8]
	alpha_0_270=lines[9]
	predicted_label_0_270=lines[10]
	weights_90_180=lines[11]
	alpha_90_180=lines[12]
	predicted_label_90_180=lines[13]
	weights_90_270=lines[14]
	alpha_90_270=lines[15]
	predicted_label_90_270=lines[16]
	weights_180_270=lines[17]
	alpha_180_270=lines[18]
	predicted_label_180_270=lines[19]

	return feature1,feature2,weights_0_90,alpha_0_90,predicted_label_0_90,weights_0_180,alpha_0_180,predicted_label_0_180,weights_0_270,alpha_0_270,predicted_label_0_270,weights_90_180,alpha_90_180,predicted_label_90_180,weights_90_270,alpha_90_270,predicted_label_90_270,weights_180_270,alpha_180_270,predicted_label_180_270


def adaboost(parameter,t_fname,model_fname):
	if(parameter=='train'):
		decision_stumps(t_fname,model_fname)

	if(parameter=='test'):
		test_adaboost(t_fname,model_fname)

	



# test_adaboost("train-datasmall.txt","model_file1.txt")
# decision_stumps("train-data.txt","model_file1.txt")
# read_model_file()
# decision_stumps()
# read_train_list("train-data.txt")
