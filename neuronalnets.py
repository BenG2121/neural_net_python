import numpy
import math
import scipy.special

class NeuralNetwork:
    def __init__(self,input_list):
        inodes = input_list[0]
        hnodes = input_list[1]
        onodes = input_list[2]
        learning_rate = input_list[3]

        self.v_test = None
        self.__w_hnou = None
        self.__w_inhn = None
        self.__v_in = None
        self.__v_hn = None
        self.__v_ou = None
        self.activation_function = numpy.vectorize(self.__activation_function)

        if isinstance(inodes,int) and inodes > 0:
            self.__inodes = inodes
        else:
            self.__inodes = None

        if isinstance(hnodes, int) and  hnodes > 0:
             self.__hnodes = hnodes
        else:
            self.__hnodes = None

        if isinstance(onodes, int) and onodes > 0:
            self.__onodes = onodes
        else:
            self.__onodes = None

        if isinstance(learning_rate, float) and learning_rate > 0 and learning_rate < 1:
            self.__learning_rate = learning_rate
        else:
            self.__learning_rate = None

    def __activation_function(self,x):
        return 1 / (1 + math.exp(-x))




    def getNumberOfXNodes(self, node_type):
        if node_type == "inodes":
            return self.__inodes
        elif node_type == "hnodes":
            return self.__hnodes
        elif node_type == "onodes":
            return self.__onodes
        else:
            return None

    def getLearningRate(self):
        return self.__learning_rate

    #w_inhn = hn*in
    #w_hnou = ou*hn
    def init_w_input_hidden(self):
        self.__w_inhn = numpy.random.rand(self.__hnodes, self.__inodes) - 0.5
        return self.__w_inhn

    def get_w_input_hidden(self):
        return self.__w_inhn

    def set_w_input_hidden(self, array):
        if array.shape == self.__w_inhn.shape:
            self.__w_inhn = array
        else:
            print("Dimensions do not match")

    def init_w_hidden_output(self):
        self.__w_hnou = numpy.random.rand(self.__onodes, self.__hnodes) - 0.5
        return self.__w_hnou

    def get_w_hidden_output(self):
        return self.__w_hnou

    def init_v_in(self):
        self.__v_in = numpy.random.rand(self.__inodes,1)*10
        return self.__v_in

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


    def get_v_in(self):
        return self.__v_in

    def init_v_hn(self,):
        self.__v_hn = numpy.zeros((self.__hnodes,1))
        return self.__v_hn

    def get_v_hn(self):
        return self.__v_hn

    def init_v_ou(self):
        self.__v_ou = numpy.zeros((self.__onodes,1))
        return self.__v_ou

    def get_v_ou(self):
        return self.__v_ou

    def print_current_values(self, prefix=""):
        print("{}v_in=\n{}\n".format(prefix,self.__v_in))
        print("{}v_hn=\n{}\n".format(prefix,self.__v_hn))
        print("{}v_ou=\n{}\n".format(prefix,self.__v_ou))

        print("{}w_input_hidden=\n{}\n".format(prefix,self.__w_inhn))
        print("{}w_hidden_output=\n{}\n".format(prefix,self.__w_hnou))

    def init_neural_network(self):
        self.init_w_input_hidden()
        self.init_w_hidden_output()
        self.init_v_in()
        self.init_v_hn()
        self.init_v_ou()

    def update_neural_network(self):
        h_in = numpy.around(self.__w_inhn.dot(self.__v_in),3)
        h_out = numpy.around(self.activation_function(h_in),3)
        o_in = numpy.around(self.__w_hnou.dot(h_out),3)
        o_out = numpy.around(self.activation_function(o_in),3)
        return o_out

if __name__ == "__main__":
    input = [3, 3, 3, 0.3]
    myNeuralNetwork = NeuralNetwork(input)
    myNeuralNetwork.init_neural_network()

    defaul_v_in_list = [0.9,0.1,0.8]
    default_v_in = numpy.array(defaul_v_in_list).reshape(len(defaul_v_in_list), 1)
    myNeuralNetwork.set_array(myNeuralNetwork.get_v_in(),default_v_in)

    default_w_inhn = numpy.array([[0.9,0.3,0.4],[0.2,0.8,0.2],[0.1,0.5,0.6]])
    myNeuralNetwork.set_array(myNeuralNetwork.get_w_input_hidden(),default_w_inhn)

    default_w_hnou = numpy.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
    myNeuralNetwork.set_array(myNeuralNetwork.get_w_hidden_output(), default_w_hnou)

    #myNeuralNetwork.print_current_values("__main__")
    output = myNeuralNetwork.update_neural_network()

    print(numpy.around(output,3))

