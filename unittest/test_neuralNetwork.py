import unittest
import numpy
import sys

sys.path.append('../src')
from neuronalnets import NeuralNetwork

class TestNeuralNetworkInit(unittest.TestCase):
    def test_instance_creation_valid_params(self):
        test_input = [6,5,4,0.4]

        myNeuralNetwork = NeuralNetwork(test_input)
        self.assertEqual(test_input[0], myNeuralNetwork.getNumberOfXNodes("inodes"))
        self.assertEqual(test_input[1], myNeuralNetwork.getNumberOfXNodes("hnodes"))
        self.assertEqual(test_input[2], myNeuralNetwork.getNumberOfXNodes("onodes"))
        self.assertEqual(test_input[3], myNeuralNetwork.getLearningRate())

    def test_instance_creation_invalid_params(self):
        test_invalid_input = [-3, 0, "-3","3","test", 0.1]
        test_learning_rate = 0.4
        for ele in test_invalid_input:
            myNeuralNetwork = NeuralNetwork([ele,ele,ele,test_learning_rate])
            self.assertEqual(None, myNeuralNetwork.getNumberOfXNodes("inodes"))
            self.assertEqual(None, myNeuralNetwork.getNumberOfXNodes("hnodes"))
            self.assertEqual(None, myNeuralNetwork.getNumberOfXNodes("onodes"))
            self.assertEqual(None, myNeuralNetwork.getNumberOfXNodes("onodes"))

    def test_instance_creation_invalid_learning_rate(self):
        test_inodes = 6
        test_hnodes = 5
        test_onodes = 4
        test_learning_rate = [-1, 0, 1, 3, "-3","3","test"]
        for ele in test_learning_rate:
            myNeuralNetwork = NeuralNetwork([test_inodes,test_hnodes,test_onodes,ele])
            self.assertEqual(None, myNeuralNetwork.getLearningRate())

    def test_invalid_getter_input(self):
        wrong_getter_input = ["abc", -1, None, 0]

        for ele in wrong_getter_input:
            myNeuralNetwork = NeuralNetwork([3, 4, 5, ele])
            self.assertEqual(None, myNeuralNetwork.getNumberOfXNodes(ele))

    def test_create_w_apis(self):
        test_input = [6,5,4,0.4]
        myNeuralNetwork = NeuralNetwork(test_input)

        w_inhn = myNeuralNetwork.init_w_input_hidden()

        for ele in numpy.nditer(w_inhn):
            self.assertTrue(ele >= -0.5 and ele <= 0.5)

        w_hnou = myNeuralNetwork.init_w_hidden_output()

        for ele in numpy.nditer(w_hnou):
            self.assertTrue(ele >= -0.5 and ele <= 0.5, msg="ele={}".format(ele))


    def test_array_sizes(self):
        input = [2,3,4,0.9]
        dim_in = input[0]
        dim_hn = input[1]
        dim_ou = input[2]

        myNeuralNetwork1 = NeuralNetwork(input)

        test_w_inhd = myNeuralNetwork1.init_w_input_hidden()
        dim_w_inhd = test_w_inhd.shape
        self.assertTrue(dim_hn == dim_w_inhd[0])
        self.assertTrue(dim_in == dim_w_inhd[1])

        test_w_hdou = myNeuralNetwork1.init_w_hidden_output()
        dim_w_hdou = test_w_hdou.shape
        self.assertTrue(dim_ou == dim_w_hdou[0])
        self.assertTrue(dim_hn == dim_w_hdou[1])

        test_v_in = myNeuralNetwork1.init_v_in()
        dim_v_in = test_v_in.shape
        self.assertTrue(dim_in == dim_v_in[0])
        self.assertTrue(1 == dim_v_in[1])

        test_v_hn = myNeuralNetwork1.init_v_hn()
        dim_v_hn = test_v_hn.shape
        self.assertTrue(dim_hn == dim_v_hn[0])
        self.assertTrue(1 == dim_v_hn[1])

        test_v_ou = myNeuralNetwork1.init_v_ou()
        dim_v_ou = test_v_ou.shape
        self.assertTrue(dim_ou == dim_v_ou[0])
        self.assertTrue(1 == dim_v_ou[1])

        #myNeuralNetwork1.print_current_values("[UNITTEST]:\n")


    def test_update_neural_network_against_values(self):
        # Create and initialize neural net
        input = [3, 3, 3, 0.3]
        myNeuralNetwork = NeuralNetwork(input)
        myNeuralNetwork.init_neural_network()

        # overwrite arrays with book values
        defaul_v_in_list = [0.9, 0.1, 0.8]
        default_v_in = numpy.array(defaul_v_in_list).reshape(len(defaul_v_in_list), 1)
        myNeuralNetwork.set_array(myNeuralNetwork.get_v_in(), default_v_in)

        default_w_inhn = numpy.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
        myNeuralNetwork.set_array(myNeuralNetwork.get_w_input_hidden(), default_w_inhn)


        default_w_hnou = numpy.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
        myNeuralNetwork.set_array(myNeuralNetwork.get_w_hidden_output(), default_w_hnou)

        # Call function under test
        output = myNeuralNetwork.update_neural_network()

        # Validate results
        expected_output = [0.726,0.708,0.778]
        expected_output_array = numpy.array(expected_output).reshape(len(expected_output), 1)

        for n in range(0,expected_output_array.shape[0]):
            for m in range(0,expected_output_array.shape[1]):
                self.assertTrue(expected_output_array[n,m] == output[n,m],msg="output value is not equal as expected!".format(expected_output,output))
if __name__ == '__main__':
    unittest.main()