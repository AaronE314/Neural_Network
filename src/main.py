#Main   
#Testing

from Neural_Network import NeuralNetwork
import numpy as np
        
def train(nn):
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    epochs = 5
    
    for e in range(epochs):
        
        print("Epoch={:d}".format(e+1))
        i=0
        for record in training_data_list:
            
            #print("record", i)
            i+=1
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            
            targets = np.zeros(output_nodes) + 0.01
            
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)
            
        test(nn)
        

def test(nn):
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    score = []
    
    print("Testing...")
    for record in test_data_list:
        
        all_values = record.split(',')
        
        correct_label = int(all_values[0])
        
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01       
            
        outputs = nn.query(inputs)
        
        label = np.argmax(outputs)
        
        if label == correct_label:
            score.append(1)
        else:
            score.append(0)
            
    score_a = np.asarray(score)
    
    print("performance = {:.2f}%".format((score_a.sum() / score_a.size)*100))

#Main
input_nodes = 784
hidden_node = 200
output_nodes = 10

learning_rate = 0.1

nn = NeuralNetwork(input_nodes, hidden_node, output_nodes, learning_rate)

train(nn)