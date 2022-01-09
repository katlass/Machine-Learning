#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:49:13 2019

@author: katelassiter
"""

import tensorflow as tf

#Define variable x
x=tf.Variable(-1.0)
#Define y within instace of GradientTape()
#Gradient tape is automatic differentiation
#TensorFlow provides the tf.GradientTape API for automatic differentiation - computing the gradient of a computation with respect to its input variables. Tensorflow "records" all operations executed inside the context of a tf.GradientTape onto a "tape". Tensorflow then uses that tape and the gradients associated with each recorded operation to compute the gradients of a "recorded" computation using reverse mode differentiation.

with tf.GradientTape() as tape:
    tape.watch(x)
    y=tf.multiply(x,x)

#y=x*2
#Get rate of change of y with repspect to x
    
#This is slope of x at a point (-1) 
g=tape.gradient(y,x)
print(g.numpy())


#Let make a graysquale vector cause yolo
#when using uniform(), it is giving mxn not the actual vectors
gray=tf.random.uniform([2,2],maxval=255,dtype='int32')
#Reshape
gray=tf.reshape(gray,[2*2,1])
#could also do, its just m (whatever its radnom when -1, and then n what you want!)
gray=tf.reshape(gray,[-1,1])

#Create color image
color= tf.random.uniform([2,2,3],maxval=255,dtype='int32')

## Reshape the grayscale image into a vector
gray_vector = reshape(gray_tensor, (-1, 1))

# Reshape the color image tensor a vector
color_vector = reshape(color_tensor, (-1, 1))

def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))




#You are given a black-and-white image of a letter, which has been encoded as a tensor, letter. You want to determine whether the letter is an X or a K. You don't have a trained neural network, but you do have a simple model, model, which can be used to classify letter.
#The 3x3 tensor, letter, and the 1x3 tensor, model, are available in the Python shell. You can determine whether letter is a K by multiplying letter by model, summing over the result, and then checking if it is equal to 1. As with more complicated models, such as neural networks, model is a collection of weights, arranged in a tensor.
# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())

#CAN use tensorflow to import data, which is useful for managing complex pipelines
#Import dataset like normal, then use np.array(df)
tf.cast(df['price'],tf.float32)


#---------------------------------------------
#---------------------------------------------
# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)

#---------------------------------------------
#---------------------------------------------
#minimize loss functions
#MSE 
#MAE
#huber error
#In statistics, the Huber loss is a loss function used in robust regression, that is less sensitive to outliers in data than the squared error loss. A variant for classification is also sometimes used.
#The Huber loss function describes the penalty incurred by an estimation procedure f. Huber (1964) defines the loss function piecewise b
#This function is quadratic for small values of a, and linear for large values, with equal values and slopes of the different sections at the two points where 
#The variable a often refers to the residuals, that is to the difference between the observed and predicted values 
#SEE GOOGLE TO DECIDE WHICH ONE
#Loss functions are accesible from tf.keras.losses()
#tf.keras.losses.mse()
#tf.keras.losses.mae()
#MSE strongly penalies outliers, high sensitiity near minimum
#MAE scales linearly with sie of error, low sensitivity near minimum
#Huber loss: similar to MSE near minimum ad similar to MAE away from min
loss = tf.keras.losses.mse(targets,predictions)
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())

# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())

#Learn about stochastic gradient descent.
#Stochastic gradient descent (often abbreviated SGD) is an iterative method for optimizing an objective function with suitable smoothness properties (e.g. differentiable or subdifferentiable). It can be regarded as a stochastic approximation of gradient descent optimization, since it replaces the actual gradient (calculated from the entire data set) by an estimate thereof (calculated from a randomly selected subset of the data).[1] Especially in big data applications this reduces the computational burden, achieving faster iterations in trade for a slightly lower convergence rate.[2]
#Adaptive Gradient Algorithm (AdaGrad) that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).
#Root Mean Square Propagation (RMSProp) that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).
#Adam realizes the benefits of both AdaGrad and RMSProp.
#Instead of adapting the parameter learning rates based on the average first moment (the mean) as in RMSProp, Adam also makes use of the average of the second moments of the gradients (the uncentered variance).
#Specifically, the algorithm calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages.
#The initial value of the moving averages and beta1 and beta2 values close to 1.0 (recommended) result in a bias of moment estimates towards zero. This bias is overcome by first calculating the biased estimates before then calculating bias-corrected estimates.

#Define a optimization operation 
opt=tf.keras.optimizers.Adam()
#will change the slope and intercept that will minimize the value of the loss

for i in range(1000):
    opt.minimize(lambda: loss_function(intercept,slope),\
    var_list=[intercept,slope])
    print(loss_function(intercept,slope))
    
print(intercept.numpy(), slope.numpy())

    
# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + slope*features

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets,predictions)

# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())

# Initialize an adam optimizer
# 0.5 is the learning rate 
opt = keras.optimizers.Adam(0.5)

for j in range(500):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept,slope), var_list=[intercept, slope])

	# Print every 100th value of the loss
	if j % 100 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)

# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)
    
    
#Batch training
for batch in pd.read_csv(,chunksize=100):
    price= np.array(batch['price'],np.float32)
    


# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the model
def linear_regression(intercept, slope, features):
	# Define the predicted values
	return intercept + slope*features

# Define the loss function
def loss_function(intercept, slope, targets,features):
	# Define the predicted values
	predictions = linear_regression(intercept, slope, features)
    
 	# Define the MSE loss
	return keras.losses.mse(targets, predictions)

# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	size_batch = np.array(batch['sqft_lot'], np.float32)

	# Extract the price values for the current batch
	price_batch = np.array(batch['price'], np.float32)

	# Complete the loss, fill in the variable list, and minimize
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())

#NEURAL NETWORKS
#A dense layer applies weights to all nodes from the previous layer

import tensorflow as tf

#Define inputs (feautures)
inputs = tf.constant([1,35])
#define weights
weights=tf.Variable([[-0.05],[-0.01]])

#define a bias which is similar to intercept
bias=tf.Variable([0.5])

#A simple dense layer
#multiply inputs by the weights
product=tf.matmult(inputs,weights)

#define dense layer
dense=tf.keras.activations.sigmid(product+bias)


#OR defining a complete model
inputs=tf.constant(data,tf.float32)
#define first dense layers
#10 equals number of nodes
dense1=tf.keras.layers.Dense(10,activation='sigmoid')(inputs)
dense2=tf.keras.layers.Dense(5, activation='sigmoid')(dense1)
outputs= tf.keras.layers.Dense(1,activation='sigmoid')(dense2)





# Initialize bias1
bias1 = Variable(1.0)

# Initialize weights1 as 3x2 variable of ones
weights1 = Variable(ones((3, 2)))

# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features,weights1)

# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)

# Print shape of dense1
print("\n dense1's output shape: {}".format(dense1.shape))

# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1,weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))
print('\n actual: 1')

# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features,weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1+bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)




# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3,activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1,activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)


#components of hidden layer
#Linear (first layer) matrix multiplication
#Nonlinear (second) activation function
#look at shape of scaterplot

import numpy as np
#Define example borrower features
young,old=0.3,0.6
low_bill,high_bill=0.1,0.5

#apply matrix multiplication step for all feature combos
young_high=1.0*young+2.0*high_bill
young_low=1.0*young+2.0*low_bill
old_high=1.0*old+2.0*high_bill
old_low=1.0*old+2.0*low_bill
#difference in default predictions for young
print(young_high-young_low)
#difference in defauly predictions for old
print(old_high-old_low)

#Now with activation function
#difference in default predictions for young
print(tf.keras.activations.sigmoid(young_high).numpy() -tf.keras.activations.sigmoid(young_low).numpy())
#difference in defauly predictions for old
print(tf.keras.activations.sigmoid(old_high).numpy() -tf.keras.activations.sigmoid(old_low).numpy())


#Change in likelihood of default is higher in younger borrowers than older borrowers


#sigmoid activation function
#-binary classification
#Low level=tf.keras.activations.sigmoid()
#high-level = sigmoid()

#ReLu 
#-hidden layers
#low-level =tf.keras.activations.relu()
#high-level=relu
#takes max of values passed to it and zero


#Softmax-used in output layer of classification problems with more than 2 classes. 
#low-level =tf.keras.activations.softmax()
#high-level=softmax
#predictaed class probabilties

iputs=tf.constant(borrower_features,tf.float32)
dense1=tf.keras.layers.Dense(16,activation='relu')(inputs)
dense2=tf.keras.Layers.Dense(8,actiation='sigmoid')(dense1)
dense3=tf.keras.Layers.Dense(4,actiation='softmax')(dense2)


#----------------
# Construct input layer from features
inputs = constant(bill_amounts,float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2,activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1,activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)

#------------------

# Construct input layer from borrower features
# Construct input layer from borrower features
inputs = constant(borrower_features,float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation="sigmoid")(inputs)


# Define second dense layer
dense2 = keras.layers.Dense(8,activation="relu")(dense1)

# Define output layer
outputs = keras.layers.Dense(6,activation="softmax")(dense2)

# Print first five predictions
print(outputs.numpy()[:5])

#------------------
#The gradient decesent optimizer 
#(stochastic)
#tf.keras.optimiers.SGD()
#learning rate

#RMS prop optimizer
#Root mean squared propagation optimier
#applied diff learning rates to each feature
#tf.keras.optimiers.RMSprop()
#learning rate
#momentum
#decay
#Allows for momentum to both build and decay
#Good for high dimension problems
#setting low value for decay paramter will prevent momentum from accumulating during learning process
#adam optimizer
#tf.keras.optimiers.Adam()
#learning rate
#beta1, increase decay by lowering beta1
#performs better with default parameter values
def model(bias,weights,feature=borrower_features):
    product=tf.matmul(features,weights)
    return tf.keras.activations.sigmoid(product+bias)

#compute predicted values and loss
def loss_function(bias,weights,targets=default,feature=borrower_features)
    predictions=model(bias,weights)
    return tf.keras.losses.binary_crossentropy(targets,predictions)

opt=tf.keras.optimiers.RMSprop(learning_rate=0.01,momentum=0.9)
opt.minimie(lambda:loss_function(bias,weights),var_list=[bias,weights])

#----------------
# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(),x_2.numpy())

#------------
#Avoiding local minima

#The previous problem showed how easy it is to get stuck in local minima. We had a simple optimization problem in one variable and gradient descent still failed to deliver the global minimum when we had to travel through local minima first. One way to avoid this problem is to use momentum, which allows the optimizer to break through local minima.

# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda:loss_function(x_2),var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())

#high momentum made it find actual min
#-----------------------------

#Random intializers
#often need to initialie thousands of variables
#tf.ones may perform badly
#so draw initial values from probability distribution
#-random  normal
#-uniform
#-glorot intializer
weights=tf.Variable(tf.random.normal([500,500]))
#vs
weights=tf.Variable(tf.random.truncated_normal([500,500]))

#vs use gloront
dense=tf.keras.layers.Dense(32,activation='relu')

#a simple solution to overfitting is applying dropout
#randomly drops weights, can ot depend on any one node
inputs=np.array(borrower_feautures,np.float32)

dense1=tf.keras.layers.Dense(32,activation='relu')(inputs)

dense2=tf.keras.layers.Dense(16,activation='relu')(dense1)

#want to drop the weights applied to 25% of the nodes randomly 
dropout1=tf.keras.layers.Dropout(0.25)(dense2) 

outputs=tf.layers.Dense(1,activation="sigmoid")(dropout1)


#--------------
# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))
2
# Define the layer 2 weights
w2 = Variable(random.normal([7,1]))

# Define the layer 2 bias
b2 = Variable(0.0)

# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)


# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)

#-----------------
#sequentiak=l api
#inout layer, hidde layer, output layer

#Defining the model
model=keras.Sequential()
#iput shape is tuple that conntains the shape of vector
model.add(keras.layer.Dense(16,activation='relu',input_shape(28*28,)))
model.add(keras.layers.Dense(8,activation='relu'))
model.add(keras.layers.Dense(4,activation='softmax'))


#compile the model
#categorical_crossentropy for classification problem with more than two classses
model.compile('adam',loss='categorical_crossentropy')

print(model.summary())


#functional api
#has set of images then et of metadata, want to restruict how they iteract in our model
model1_inputs=tf.keras.Input(shape=(28*28,))
model2_inputs=tf.keras.Input(shape=(10,))

model1_layer1=tf.keras.layers.Dense(12,activation='relu')(model1_inputs)
model1_layer2=tf.keras.layers.Dense(4,activation='softmax')(model1_layer1)



model2_layer1=tf.keras.layers.Dense(8,activation='relu')(model2_inputs)
model2_layer2=tf.keras.layers.Dense(4,activation='softmax')(model2_layer1)

#merge model 1 and 2
merged = tf.keras.layers.add([model1_layer2,model2_layer2])

#define a functional model
model=tf.keras.Model(inputs=[model1_inputs,model2_inputs],outputs=merged)
model.compile('adam',loss='categorical_crossentropy')
#------------------
# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8,activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4,activation='softmax'))

# Print the model architecture
print(model.summary())
#------------
# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4,activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())
#------------
# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

#------------

#Trainig and validating
#load clean data
#define mode'
#train model
#evaluate

model=tf.keras.Sequential()
#784 because contains 28 by 28 images reshaped into vectors
model.add(tf.keras.layers.Dense(16,activation='relu',input_shape=(784,)))

model.add(tf.keras.layers.Dense(4,actiation='softmax'))



#batch_size=32, epochs (number of times you train on the full set of batches. this lets you revissit the sample with new weights and optimization parameters!),
#(validation_split divides the data into two parts, train and validation set.
#validation=0.2 mean putting 20% of the data in the validation set
model.fit(image_features,image_labels)

#can do this
model.compile('adam',loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(image_features,image_labels)

#split off a genuine test set not used in the training
model.evaluate(test)


#---------------------------
# Define a sequential model
model=keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4,activation="softmax"))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features,sign_language_labels, epochs=5)
#---------------------------

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)
#---------------------------
# Define sequential model
model=keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024,activation='relu',input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=200, validation_split=.5)

#Excellent work! You may have noticed that the validation loss, val_loss, was substantially higher than the training loss, loss. Furthermore, if val_loss started to increase before the training process was terminated, then we may have overfitted. When this happens, you will want to try decreasing the number of epochs.
#---------------------------
# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test =  small_model.evaluate(test_features,test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features,test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))
#Small - Train: 0.1618082270026207, Test: 0.2816877818107605
#Large - Train: 0.03647002600133419, Test: 0.14837740659713744

#Great job! Notice that the gap between the test and train set losses is substantially higher for large_model, suggesting that overfitting may be an issue. Furthermore, both test and train set performance is better for large_model. This suggests that we may want to use large_model, but reduce the number of training epochs.
#_----------------------
#Training models with the Estimators API
#is a high level submodule in tf, less flexible than keras, enf  orces set of best practices by enforcing restrictions, fatser deployment (less code), many premade models
#-define feature columns, load/tranform data which is dictionary object, define estimators,apply train operations

size=tf.feature_column.numeric_column("size")
rooms=tf.feature_column.categorical_column_with_vocabulary_list("rooms",["1","2","3","4","5"])
#merge these into a list f feature columns
features_list=[size,rooms]

#define input data function
def input_fn():
    #define feature dictionary
    features={"size":[1340,1690,2720],"rooms":[1,3,4]}
    #define labels
    labels=[221900,538000,180000]
    return features,labels

#define deep neural network regression 
model0 = tf.estimator.DNNRegressor(feature_columns=features_list,hidden_units=[10,6,6,3])
#DNNRegressor allows us to predict a continuous target
#hidden_units is nodes
model0.train(input_fn,steps=20)

#if we instead wanted to do classiifcation
model1 = tf.estimator.DNNClassifier(feature_columns=features_list,hidden_units=[32,16,8],n_classes=4)
model1.train(input_fn,steps=20)

##---------------------------------
# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels

# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)



##############################################################
#ADVANCED#
from tensorflow.keras.layers import Input
input_tensor = Input(shape=(1,))
from tensorflow.keras.layers import Dense
output_layer=Dense(1)
output_tensor=output_layer(input_tensor)
#-------------
# Import Input from keras.layers
from keras.layers import Input

# Create an input layer of shape 1
input_tensor = Input(shape=(1,))
# Load layers
from keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape=(1,))

# Dense layer
output_layer = Dense(1)

# Connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)


#----
# Load layers
from keras.layers import Input,Dense

# Input layer
input_tensor = Input(shape=(1,))

# Create a dense layer and connect the dense layer to the input_tensor in one step
# Note that we did this in 2 steps in the previous exercise, but are doing it in one step now
output_tensor = Dense(1)(input_tensor)

#------------
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)
from keras.models import Model
model=Model(input_tensor,output_tensor)
model.compile(optimier='adam',loss='mae')
model.summary()
#visualize
output_layer=Dense(1,name='Predicted-Score-Diff')
output_tensor=output_layer(input_tensor)
model=Model(input_tensor,output_tensor)
plot_model(model,to_file='model.png')
from matplotlib import pyplot as plt
img=plt.imread('model.png')
plt.imshow(img)
plt.show()

#-------
# Input/dense/output layers
from keras.layers import Input, Dense
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)

# Build the model
from keras.models import Model
model = Model(input_tensor, output_tensor)
# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
# Import the plotting function
from keras.utils import plot_model
import matplotlib.pyplot as plt

# Summarize the model
model.summary()

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()

#---------------
import pandas as pd
df=pd.read_csv("...")
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(df['seed-diff'],df['score-diff'],batch_size=64,validation_split=.20,verbose=True)
#batch sie shows how many rows are used for stoichastic gradient descent
#verbose prints a log during training
model.evaluate(df['seed-diff'],df['score-diff'])
#--------
#epochs is walks over a dataset
# Now fit the model
model.fit(games_tourney_train['seed_diff'], games_tourney_train['score_diff'],
          epochs=1,
          batch_size=128,
          validation_split=0.1,
          verbose=True)

# Load the X variable from the test data
X_test = games_tourney_test['seed_diff']

# Load the y variable from the test data
y_test = games_tourney_test['score_diff']

# Evaluate the model on the test data
print(model.evaluate(X_test, y_test, verbose=False))

#------------------------------------------
#category embeddings advanced layers only available in deep learning
#good for lots of variables on categorical data 
#know we are trying to evaluate the entire worth of a single team
input_tensor = Input(shape=(1,)) #will be the number of each team (a unique id)
from keras.layers import Embedding
n_teams = 10887 #number teams
embed_layer= Embedding(input_dim=n_teams, input_length=1,output_dim=1,name="Team-Strength-Lookup")
#input_length = 1 because to identify the team we are using a single integer, unique team id
#want to produce a single team strength rating
embed_tensor = embed_layer(input_tensor)

#embedded layers increase the dimensionality of your data
#your data has 2 dims, rows and columns, the embeded layers add a 3rd dim (very relavent for images and text)
#in this case we dont need 3d so we use flatten to make it 2d
from keras.layers import Flatten 
flatten_tensor = Flatten()(embed_tensor)
#the flatten layer is the output layer for the imbedding process
#flattening can be used to transform data from multiple dimensions back down to 2
#useful for image, text and time series data
model = Model(input_tensor,flatten_tensor)

#--------------------------------
# Imports
from keras.layers import Embedding
from numpy import unique

# Count the unique number of teams
n_teams = unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')

# Imports
from keras.layers import Input, Embedding, Flatten
from keras.models import Model

# Create an input layer for the team ID
teamid_in = Input(shape=(1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')
#------------------
#Shared layers
#a model with 2 inputs, one for each team in playing a game
#want each team to use same embedding layer i e shared layer
#apply exact same weights on two diff obs

input_tensor_1 = Input(shape=(1,))
input_tensor_2 = Input(shape=(1,))
#Dense is a function that takes a tensoras input and releases a tensor as outout
shared_layer=Dense(1)
output_tensor_1=shared_layer(input_tensor_1)
output_tensor_2=shared_layer(input_tensor_2)


#=====================
input_tensor = Input(shape=(1,)) #will be the number of each team (a unique id)
from keras.layers import Embedding
n_teams = 10887 #number teams
embed_layer= Embedding(input_dim=n_teams, input_length=1,output_dim=1,name="Team-Strength-Lookup")
#input_length = 1 because to identify the team we are using a single integer, unique team id
#want to produce a single team strength rating
embed_tensor = embed_layer(input_tensor)
#embedded layers increase the dimensionality of your data
#your data has 2 dims, rows and columns, the embeded layers add a 3rd dim (very relavent for images and text)
#in this case we dont need 3d so we use flatten to make it 2d
from keras.layers import Flatten 
flatten_tensor = Flatten()(embed_tensor)
#the flatten layer is the output layer for the imbedding process
#flattening can be used to transform data from multiple dimensions back down to 2
#useful for image, text and time series data
model = Model(input_tensor,flatten_tensor)
input_tensor_1 = Input(shape=(1,))
input_tensor_2 = Input(shape=(1,))
output_tensor_1=model(input_tensor_1)
output_tensor_2=model(input_tensor_2)
#this will use same model, same layers, same weight for mapping each input to its corresponding output
#takes arbitrary number of keras layers and wrap them up into a model
#reuse that model to share that sequence of steps for different input layers
#----------------------------
# Load the input layer from keras.layers
from keras.layers import Input

# Input layer for team 1
team_in_1 = Input((1,),name = "Team-1-In")

# Separate input layer for team 2
team_in_2 = Input((1,),name = "Team-2-In")
# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)
#-------
#now that you have multiple inputs and a shared layer you need to combine your inputs into a single layer
#you can use to predict a single output
#merge layers allow you to define advanced nonsequential networked apologies?
#Merge Layers 
#-add
#subtrrct
#mutlipy #---ABOVE SAME SHAPE LAYERS
#concatenate #---APPEND THE TWO LAYERS TOGETHER

in_tensor_1 = Input((1,))
in_tensor_2 = Input((1,))
out_tensor=Add()([in_tensor_1,in_tensor_2])
model=Model([in_tensor_1,in_tensor_2],out_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')


#---------
# Import the Subtract layer from keras
from keras.layers import Subtract

# Create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# Imports
from keras.layers import Subtract
from keras.models import Model

# Subtraction layer from previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# Create the model
model = Model([team_in_1, team_in_2], score_diff)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

#----------
#fitting and predicting with multiple inputs
model.fit([data1,data2],target)
model.predict([np.array([[1]]),np.array(([2]))])
>>array([[3.]])
model.predict([np.array([[42]]),np.array(([119]))])
>>array([[161.]])

model.evaluate([np.array([[-1]]),np.array([[-2]]),np.array((-3))])

#=-----------
# Get the team_1 column from the regular season data
input_1 = games_season['team_1']

# Get the team_2 column from the regular season data
input_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_1,input_2],
          games_season['score_diff'],
          epochs=1,
          batch_size=2048,
          validation_split=0.1,
          verbose=True)

# Get team_1 from the tournament data
input_1 = games_tourney['team_1']

# Get team_2 from the tournament data
input_2 = games_tourney["team_2"]

# Evaluate the model using these inputs
print(model.evaluate([input_1, input_2], games_tourney['score_diff'], verbose=False))
#_----------------------------
#THREE INPUT MODELS
in_tensor_1 = Input((1,))
in_tensor_2 = Input((1,))
in_tensor_3 = Input((1,))
out_tensor=Concatenate()([in_tensor_1,in_tensor_2,in_tensor_3])
output_tensor=Dense(1)(out_tensor)
model=Model([in_tensor_1,in_tensor_2,in_tensor_3],out_tensor)
#-----
#with a shared layer
shared_layer = Dense(1)
shared_tensor_1 = shared_layer(in_tensor_1)
shared_tensor_2 = shared_layer(in_tensor_1)
out_tensor=Concatenate()([shared_tensor_1,shared_tensor_2,in_tensor_3])
out_tensor=Dense(1)(out_tensor)
model=Model([in_tensor_1,in_tensor_2,in_tensor_3],out_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit([[train['col1'],train['col2'],train['col3']],train_data['target'])
model.evaluate([test['col1'],test['col2'],test['col3']],test['target'])

#-------
# Create an Input for each team
team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')

# Create an input for home vs away
home_in = Input(shape=(1,), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)

# Import the model class
from keras.models import Model

# Make a Model
model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Fit the model to the games_season dataset
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs=1,
          verbose= True,
          validation_split=0.1,
          batch_size=2048)

# Evaluate the model on the games_tourney dataset
print(model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']], games_tourney['score_diff'], verbose=False))

#-----------------
#stacking models
#model stacking is when you use the predictions from one model as the input to another model
#most sophistacted way of combining models

#if all inputs numeric can use
in_tensor=Input(share=(3,))
out_tensor=Dense(1)(in_tensor)
model=Model(in_tensor,out_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')
train_x = train_data[['home','seed_diff','pred']]
train_y = train_data['score_diff']
model.fit(train_x,train_y,epochs=10,validation_split =.1)
test_X = test_data[['home','seed_diff','pred']]
test_y = test_data['score_diff']
model.evaluate(test_X,test_y)
#--------
# Predict
games_tourney['pred'] = model.predict([games_tourney['team_1'],
                                             games_tourney['team_2'],
                                             games_tourney['home']])

#---
    # Create an input layer with 3 columns
input_tensor = Input((3,))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
# Fit the model
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs=1,
          verbose=True)
# Evaluate the model on the games_tourney_test dataset
print(model.evaluate(games_tourney_test[['home','seed_diff','prediction']],
               games_tourney_test['score_diff'], verbose=False))
#-------------
#Two output Models
#Use the model to predict both the scores of the teams
input_tensor = Input(shapre=(1,))
output_tensor=Dense(2)(input_tensor)
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(games_tourney_train[['seed-diff']],games_tourney_train[['score_1','score_2']],epochs=500)
#take a look at what the model learned
model.get_weights()
returns [array([[0.607,-0.59800]],dtype=float32),array([[70.394,70.393]],dtype=float32)]
#These are the slopes and intercepts of the model!
model.evaluate(games_tourney_test[['seed-diff']],games_tourney_test[['score_1','score_2']])
#reasonably good model
#--------------
# Define the input
input_tensor = Input((2,))

# Define the output
output_tensor = Dense(2)(input_tensor)

# Create a model
model = Model(input_tensor,output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
# Fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']],
  		  games_tourney_train[['score_1', 'score_2']],
  		  verbose=True,
  		  epochs=100,
  		  batch_size=16384)

# Print the model's weights
print(model.get_weights())

# Print the column means of the training data
print(games_tourney_train.mean())

# Evaluate the model on the tournament test data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']], games_tourney_test[['score_1', 'score_2']], verbose=False))
#--------
#Single model for classification and regression
input_tensor = Input((1,))
output_tensor_reg = Dense(1)(input_tensor)
#using the regression model prediction as input and then add another dense layer on top of it
#this will map the predicted score differences to probabailties that team 1 wins
output_tensor_class = Dense(1,activation='sigmoid')(output_tensor_reg)
model = Model(input_tensor,[output_tensor_reg,output_tensor_class])
model.compile(loss=['mean_absolute_error','binary_crossentropy'],optimizer='adam')
X=games_tourney_train[['seed_diff']]
y_reg=games_tourney_train[['score_diff']]
y_class=games_tourney_train[['won']]
model.fit(X,[y_reg,y_classs],epochs=100)
model.get_weights()
returns [array([[1.23]],dtype=float32),array([-0.05],dtype=float32),array([[0.13]],dtype=float32),array([0.007],dtype=float32)]
#weight of 1.24 means 1.24 score diff with seed diff of 1
#the win or lose is sigmoid though so you have to put it in a function
from scipy.special import expit as sigmoid
print(sigmoid(1*0.13+0.007))
returns 0.53
#probability they win is 53%
games_tourney_test[['seed_diff']]
y_reg=games_tourney_test[['score_diff']]
y_class=games_tourney_test[['won']]
model.evaluate(X,[y_reg,y_class])

returns [9.8666,9.28,0.58]
#first number is loss function used by the model (sum of all the output losses), second is loss part of model for regression, last part is loss for the classification
#so has MAE of 9.28 and a log loss of 0.58
#-------------------
# Inputs (seed difference and predicted score difference) have a mean of very close to zero, and outputs both have means that are close to zero, so your model shouldn't need the bias term to fit the data well.
# Create an input layer with 2 columns
input_tensor = Input((2,))

# Create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# Create the second output (use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)

# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1 , output_tensor_2])

# Import the Adam optimizer
from keras.optimizers import Adam

# Compile the model with 2 losses and the Adam optimzer with a higher learning rate
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(0.01))

# Fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney_train[['seed_diff', 'pred']],
          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          epochs=10,
          verbose=True,
          batch_size=16384)
# Print the model weights
print(model.get_weights())

# Print the training data means
print(games_tourney_train.mean())

# Import the sigmoid function from scipy
from scipy.special import expit as sigmoid

# Weight from the model
weight = 0.14

# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))

# Evaluate the model on new data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']],
               [games_tourney_test[['score_diff']], games_tourney_test[['won']]], verbose=False))
#wrap up
#share layers are very useful when you want to compare two things
-baskeyball teams
-image similarity/retrieval
#document similarity
#also known as siamese networks
#for text use embedded layer and LSTM then concat then output
#for numeric go right to concat and ouput
#for images do convolution to colvulution to concat layer to output
#----
#side note
#skip connections
#helps find the globalminimum of the loss function for adam optimizer?
input_tensor = Input((100,))
hidden_tensor = Dense(256,activation='relu')(input_tensor)
hidden_tensor = Dense(256,activation='relu')(hidden_tensor)
hidden_tensor = Dense(256,activation='relu')(hidden_tensor)
#use concatenate to concateate inputs to the deep network output right before the final output layer
output_tensor=Concatenate()([input_tensor,hidden_tensor])
output_tensor=Dense(256,activation='relu')(output_tensor)

output_tensor = 
]



