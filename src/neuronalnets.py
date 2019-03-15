import numpy
import math

class NeuralNetwork:
    def __init__(self,input_list: list):
        self.activation_function = numpy.vectorize(self.__activation_function)

        self.inputnodes = self.isvalidnode(input_list[0])
        self.hiddennodes = self.isvalidnode(input_list[1])
        self.outputnodes = self.isvalidnode(input_list[2])
        self.learning_rate = self.isvalidrate(input_list[3])

    def __activation_function(self,x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def isvalidnode(par):
        if isinstance(par, int) and par > 0:
            return par
        else:
            return None

    @staticmethod
    def isvalidrate(par):
        if isinstance(par, float) and par > 0 and par < 1:
            return par
        else:
            return None

    def get_number_of_inputnodes(self):
        return self.inputnodes

    def get_number_of_hiddennodes(self):
        return self.hiddennodes

    def get_number_of_outputnodes(self):
        return self.outputnodes

    def get_learning_rate(self):
        return self.learning_rate

    def get_weight_input_hidden(self):
        return self.weight_input_hidden

    def get_weight_hidden_output(self):
        return self.weight_hidden_output

    def get_input_vector(self):
        return self.input_vector

    def get_hidden_vector(self):
        return self.hidden_vector

    def get_output_vector(self):
        return self.output_vector

    def init_input_vector(self):
        self.input_vector = numpy.random.rand(self.inputnodes,1)*10
        return self.input_vector

    #weight_input_hidden = hn*in
    #weight_hidden_output = ou*hn
    def init_weight_input_hidden(self):
        self.weight_input_hidden = numpy.random.rand(self.hiddennodes, self.inputnodes) - 0.5
        return self.weight_input_hidden

    def init_weight_hidden_output(self):
        self.weight_hidden_output = numpy.random.rand(self.outputnodes, self.hiddennodes) - 0.5
        return self.weight_hidden_output


    def set_weight_input_hidden(self, array):
        if array.shape == self.weight_input_hidden.shape:
            self.weight_input_hidden = array
        else:
            print("Dimensions do not match")

    def init_hidden_vector(self,):
        self.hidden_vector = numpy.zeros((self.hiddennodes,1))
        return self.hidden_vector

    def init_output_vector(self):
        self.output_vector = numpy.zeros((self.outputnodes,1))
        return self.output_vector

    def init_neural_network(self):
        self.init_weight_input_hidden()
        self.init_weight_hidden_output()
        self.init_input_vector()
        self.init_hidden_vector()
        self.init_output_vector()

    def update_neural_network(self):
        hidden_in = numpy.around(self.weight_input_hidden.dot(self.input_vector),3)
        hidden_out = numpy.around(self.activation_function(hidden_in),3)
        output_in = numpy.around(self.weight_hidden_output.dot(hidden_out),3)
        output_out = numpy.around(self.activation_function(output_in),3)
        return output_out

    def set_array(self, v1,v2):
        try:
            if v1.shape == v2.shape:
                for n in range (0,v1.shape[0]):
                    for m in range (0,v1.shape[1]):
                        v1[n,m] = v2[n,m]
            else:
                print("Dimensions do not match")
        except:
            print("Dimensions dont match")

    def print_current_values(self, prefix=""):
        print("{}input_vector=\n{}\n".format(prefix,self.input_vector))
        print("{}hidden_vector=\n{}\n".format(prefix,self.hidden_vector))
        print("{}output_vector=\n{}\n".format(prefix,self.output_vector))

        print("{}weight_input_hidden=\n{}\n".format(prefix,self.weight_input_hidden))
        print("{}weight_hidden_output=\n{}\n".format(prefix,self.weight_hidden_output))

if __name__ == "__main__": # pragma: no cover
    input = [3, 3, 3, 0.3]
    myNeuralNetwork = NeuralNetwork(input)
    myNeuralNetwork.init_neural_network()

    default_input_vector_list = [0.9,0.1,0.8]
    default_input_vector = numpy.array(default_input_vector_list).reshape(len(default_input_vector_list), 1)
    myNeuralNetwork.set_array(myNeuralNetwork.get_input_vector(),default_input_vector)

    default_weight_input_hidden = numpy.array([[0.9,0.3,0.4],[0.2,0.8,0.2],[0.1,0.5,0.6]])
    myNeuralNetwork.set_array(myNeuralNetwork.get_weight_input_hidden(),default_weight_input_hidden)

    default_weight_hidden_output = numpy.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
    myNeuralNetwork.set_array(myNeuralNetwork.get_weight_hidden_output(), default_weight_hidden_output)

    #myNeuralNetwork.print_current_values("__main__")
    output = myNeuralNetwork.update_neural_network()

    print(numpy.around(output,3))

