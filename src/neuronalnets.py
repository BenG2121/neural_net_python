import numpy
import math

class NeuralNetwork:
    ROUND_VALUE = 3
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

    def query(self, _input_list):
        if type(_input_list) is list:
            input_list = numpy.array(_input_list, ndmin=2).T
        else:
            input_list = _input_list

        hidden_in = numpy.around(self.weight_input_hidden.dot(input_list), self.ROUND_VALUE)
        hidden_out = numpy.around(self.activation_function(hidden_in), self.ROUND_VALUE)
        output_in = numpy.around(self.weight_hidden_output.dot(hidden_out), self.ROUND_VALUE)
        output_out = numpy.around(self.activation_function(output_in), self.ROUND_VALUE)
        return output_out

    def train_network(self, _input_list, _target_lists):
        target = numpy.array(_target_lists, ndmin=2).T
        inputs = numpy.array(_input_list, ndmin=2).T

        hidden_in = numpy.around(self.weight_input_hidden.dot(inputs),self.ROUND_VALUE)
        hidden_out = numpy.around(self.activation_function(hidden_in),self.ROUND_VALUE)
        output_in = numpy.around(self.weight_hidden_output.dot(hidden_out),self.ROUND_VALUE)
        output_out = numpy.around(self.activation_function(output_in),self.ROUND_VALUE)

        self.error_output_vector = target - output_out
        self.error_hidden_vector = numpy.dot(self.weight_hidden_output.T, self.error_output_vector)

        self.weight_hidden_output += self.learning_rate * \
                                     numpy.around(numpy.dot(self.error_output_vector * output_out * (1.0 - output_out),
                                                            hidden_out.T),self.ROUND_VALUE)

        self.weight_input_hidden += self.learning_rate * \
                                 numpy.around(numpy.dot(self.error_hidden_vector * hidden_out * (1.0 - hidden_out),
                                                        inputs.T), self.ROUND_VALUE)

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
    output = myNeuralNetwork.query(myNeuralNetwork.input_vector)

    print(numpy.around(output,3))

