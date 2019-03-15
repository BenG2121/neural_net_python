import unittest
import numpy
import sys

sys.path.append('../src')
from neuronalnets import NeuralNetwork

class TestNeuralNetworkInit(unittest.TestCase):
    def test_instance_creation_valid_params(self):
        test_input = [6,5,4,0.4]

        myNeuralNetwork = NeuralNetwork(test_input)
        self.assertEqual(test_input[0], myNeuralNetwork.get_number_of_inodes())
        self.assertEqual(test_input[1], myNeuralNetwork.get_number_of_hnodes())
        self.assertEqual(test_input[2], myNeuralNetwork.get_number_of_onodes())
        self.assertEqual(test_input[3], myNeuralNetwork.get_learning_rate())

    def test_instance_creation_invalid_params(self):
        test_invalid_input = [-3, 0, "-3","3","test", 0.1]
        test_learning_rate = 0.4
        for ele in test_invalid_input:
            myNeuralNetwork = NeuralNetwork([ele,ele,ele,test_learning_rate])
            self.assertEqual(None, myNeuralNetwork.get_number_of_inodes())
            self.assertEqual(None, myNeuralNetwork.get_number_of_hnodes())
            self.assertEqual(None, myNeuralNetwork.get_number_of_onodes())
            self.assertEqual(test_learning_rate, myNeuralNetwork.get_learning_rate())

    def test_instance_creation_invalid_learning_rate(self):
        test_inodes = 6
        test_hnodes = 5
        test_onodes = 4
        test_learning_rate = [-1, 0, 1, 3, "-3","3","test"]
        for ele in test_learning_rate:
            myNeuralNetwork = NeuralNetwork([test_inodes,test_hnodes,test_onodes,ele])
            self.assertEqual(None, myNeuralNetwork.get_learning_rate())

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

    def test_input_vector_init_and_getter(self):
        """"Input vector is initialized with random values. Expected values are not 0, not 0.0 and not None"""
        input = [3, 3, 3, 0.3]
        myNeuralNetwork = NeuralNetwork(input)

        myNeuralNetwork.init_v_in()
        v_in_tmp = myNeuralNetwork.get_v_in()
        for ele in v_in_tmp:
            self.assertNotEqual(ele, 0.0)
            self.assertNotEqual(ele, 0)
            self.assertNotEqual(ele, None)

    def test_hidden_vector_init_and_getter(self):
        """"Hidden vector is initialized with zeros. Expected values are 0, not None"""
        input = [3, 3, 3, 0.3]
        myNeuralNetwork = NeuralNetwork(input)

        myNeuralNetwork.init_v_hn()
        v_hn_tmp = myNeuralNetwork.get_v_hn()
        for ele in v_hn_tmp:
            self.assertEqual(ele, 0)
            self.assertNotEqual(ele, None)

    def test_output_vector_init_and_getter(self):
        """"Output vector is initialized with zeros. Expected values are 0, not None"""
        input = [3, 3, 3, 0.3]
        myNeuralNetwork = NeuralNetwork(input)

        myNeuralNetwork.init_v_ou()
        v_ou_tmp = myNeuralNetwork.get_v_ou()
        for ele in v_ou_tmp:
            self.assertEqual(ele, 0)
            self.assertNotEqual(ele, None)

if __name__ == '__main__':
    unittest.main()