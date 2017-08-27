import numpy
import scipy.special
import scipy.ndimage
import matplotlib.pyplot
%matplotlib inline
class neuralNetwork:
    #initial 
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #set lr
        self.lr = learningrate
        #link weight matrices wih & who
        self.wih = numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        #sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    #train
    def train(self,inputs_list,targets_list):
        #
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        #
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        #
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #error
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        #update link weights
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass
    #query
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
pass

#set number
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
#set lr
learning_rate = 0.01
#create object
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open('mnist_dataset/mnist_train.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        #do for rotate
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        inputs_plus10 = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28),10,cval = 0.01,reshape = False)
        inputs_minus10 = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28),-10,cval = 0.01,reshape = False)
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        n.train(inputs_plus10.reshape(784),targets)
        n.train(inputs_minus10.reshape(784),targets)
        pass
    pass
test_data_file = open('mnist_dataset/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
