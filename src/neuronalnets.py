import numpy
import math
import scipy.special

class NeuralNetwork:
    def __init__(self,input_list: list):
        inodes = input_list[0]
        hnodes = input_list[1]
        onodes = input_list[2]
        learning_rate = input_list[3]

        self.v_test = None
        self.w_hnou = None
        self.w_inhn = None
        self.v_in = None
        self.v_hn = None
        self.v_ou = None
        self.activation_function = numpy.vectorize(self.__activation_function)

        self.inodes = self.isvalidnode(inodes)
        self.hnodes = self.isvalidnode(hnodes)
        self.onodes = self.isvalidnode(onodes)
        self.learning_rate = self.isvalidrate(learning_rate)

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

    def getNumberOfXNodes(self, node_type):
        if node_type == "inodes":
            return self.inodes
        elif node_type == "hnodes":
            return self.hnodes
        elif node_type == "onodes":
            return self.onodes
        else:
            return None

    def getLearningRate(self):
        return self.learning_rate

    #w_inhn = hn*in
    #w_hnou = ou*hn
    def init_w_input_hidden(self):
        self.w_inhn = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        return self.w_inhn

    def get_w_input_hidden(self):
        return self.w_inhn

    def set_w_input_hidden(self, array):
        if array.shape == self.w_inhn.shape:
            self.w_inhn = array
        else:
            print("Dimensions do not match")

    def init_w_hidden_output(self):
        self.w_hnou = numpy.random.rand(self.onodes, self.hnodes) - 0.5
        return self.w_hnou

    def get_w_hidden_output(self):
        return self.w_hnou

    def init_v_in(self):
        self.v_in = numpy.random.rand(self.inodes,1)*10
        return self.v_in

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
        return self.v_in

    def init_v_hn(self,):
        self.v_hn = numpy.zeros((self.hnodes,1))
        return self.v_hn

    def get_v_hn(self):
        return self.v_hn

    def init_v_ou(self):
        self.v_ou = numpy.zeros((self.onodes,1))
        return self.v_ou

    def get_v_ou(self):
        return self.v_ou

    def print_current_values(self, prefix=""):
        print("{}v_in=\n{}\n".format(prefix,self.v_in))
        print("{}v_hn=\n{}\n".format(prefix,self.v_hn))
        print("{}v_ou=\n{}\n".format(prefix,self.v_ou))

        print("{}w_input_hidden=\n{}\n".format(prefix,self.w_inhn))
        print("{}w_hidden_output=\n{}\n".format(prefix,self.w_hnou))

    def init_neural_network(self):
        self.init_w_input_hidden()
        self.init_w_hidden_output()
        self.init_v_in()
        self.init_v_hn()
        self.init_v_ou()

    def update_neural_network(self):
        h_in = numpy.around(self.w_inhn.dot(self.v_in),3)
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

