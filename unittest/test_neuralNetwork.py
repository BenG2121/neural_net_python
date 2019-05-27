import unittest
import numpy
import sys

sys.path.append('../src')
from neuronalnets import NeuralNetwork

class TestNeuralNetworkInit(unittest.TestCase):
    def test_instance_creation_valid_params(self):
        test_input = [6,5,4,0.4]

        myNeuralNetwork = NeuralNetwork(test_input)
        self.assertEqual(test_input[0], myNeuralNetwork.get_number_of_inputnodes())
        self.assertEqual(test_input[1], myNeuralNetwork.get_number_of_hiddennodes())
        self.assertEqual(test_input[2], myNeuralNetwork.get_number_of_outputnodes())
        self.assertEqual(test_input[3], myNeuralNetwork.get_learning_rate())

    def test_instance_creation_invalid_params(self):
        test_invalid_input = [-3, 0, "-3","3","test", 0.1]
        test_learning_rate = 0.4
        for ele in test_invalid_input:
            myNeuralNetwork = NeuralNetwork([ele,ele,ele,test_learning_rate])
            self.assertEqual(None, myNeuralNetwork.get_number_of_inputnodes())
            self.assertEqual(None, myNeuralNetwork.get_number_of_hiddennodes())
            self.assertEqual(None, myNeuralNetwork.get_number_of_outputnodes())
            self.assertEqual(test_learning_rate, myNeuralNetwork.get_learning_rate())

    def test_instance_creation_invalid_learning_rate(self):
        test_inputnodes = 6
        test_hiddennodes = 5
        test_outputnodes = 4
        test_learning_rate = [-1, 0, 1, 3, "-3","3","test"]
        for ele in test_learning_rate:
            myNeuralNetwork = NeuralNetwork([test_inputnodes,test_hiddennodes,test_outputnodes,ele])
            self.assertEqual(None, myNeuralNetwork.get_learning_rate())

    def test_create_w_apis(self):
        test_input = [6,5,4,0.4]
        myNeuralNetwork = NeuralNetwork(test_input)

        weight_input_hidden = myNeuralNetwork.init_weight_input_hidden()

        for ele in numpy.nditer(weight_input_hidden):
            self.assertTrue(ele >= -0.5 and ele <= 0.5)

        weight_hidden_output = myNeuralNetwork.init_weight_hidden_output()

        for ele in numpy.nditer(weight_hidden_output):
            self.assertTrue(ele >= -0.5 and ele <= 0.5, msg="ele={}".format(ele))


    def test_array_sizes(self):
        input = [2,3,4,0.9]
        dim_in = input[0]
        dim_hn = input[1]
        dim_ou = input[2]

        myNeuralNetwork1 = NeuralNetwork(input)

        test_w_inhd = myNeuralNetwork1.init_weight_input_hidden()
        dim_w_inhd = test_w_inhd.shape
        self.assertTrue(dim_hn == dim_w_inhd[0])
        self.assertTrue(dim_in == dim_w_inhd[1])

        test_w_hdou = myNeuralNetwork1.init_weight_hidden_output()
        dim_w_hdou = test_w_hdou.shape
        self.assertTrue(dim_ou == dim_w_hdou[0])
        self.assertTrue(dim_hn == dim_w_hdou[1])

        test_input_vector = myNeuralNetwork1.init_input_vector()
        dim_input_vector = test_input_vector.shape
        self.assertTrue(dim_in == dim_input_vector[0])
        self.assertTrue(1 == dim_input_vector[1])

        test_hidden_vector = myNeuralNetwork1.init_hidden_vector()
        dim_hidden_vector = test_hidden_vector.shape
        self.assertTrue(dim_hn == dim_hidden_vector[0])
        self.assertTrue(1 == dim_hidden_vector[1])

        test_output_vector = myNeuralNetwork1.init_output_vector()
        dim_output_vector = test_output_vector.shape
        self.assertTrue(dim_ou == dim_output_vector[0])
        self.assertTrue(1 == dim_output_vector[1])

        #myNeuralNetwork1.print_current_values("[UNITTEST]:\n")


    def test_update_neural_network_against_values(self):
        # Create and initialize neural net
        input = [3, 3, 3, 0.3]
        myNeuralNetwork = NeuralNetwork(input)
        myNeuralNetwork.init_neural_network()

        # overwrite arrays with book values
        default_input_vector_list = [0.9, 0.1, 0.8]
        default_input_vector = numpy.array(default_input_vector_list).reshape(len(default_input_vector_list), 1)
        myNeuralNetwork.set_array(myNeuralNetwork.get_input_vector(), default_input_vector)

        default_weight_input_hidden = numpy.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
        myNeuralNetwork.set_array(myNeuralNetwork.get_weight_input_hidden(), default_weight_input_hidden)


        default_weight_hidden_output = numpy.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
        myNeuralNetwork.set_array(myNeuralNetwork.get_weight_hidden_output(), default_weight_hidden_output)

        # Call function under test
        output = myNeuralNetwork.query(myNeuralNetwork.input_vector)

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

        myNeuralNetwork.init_input_vector()
        input_vector_tmp = myNeuralNetwork.get_input_vector()
        for ele in input_vector_tmp:
            self.assertNotEqual(ele, 0.0)
            self.assertNotEqual(ele, 0)
            self.assertNotEqual(ele, None)

    def test_hidden_vector_init_and_getter(self):
        """"Hidden vector is initialized with zeros. Expected values are 0, not None"""
        input = [3, 3, 3, 0.3]
        myNeuralNetwork = NeuralNetwork(input)

        myNeuralNetwork.init_hidden_vector()
        hidden_vector_tmp = myNeuralNetwork.get_hidden_vector()
        for ele in hidden_vector_tmp:
            self.assertEqual(ele, 0)
            self.assertNotEqual(ele, None)

    def test_output_vector_init_and_getter(self):
        """"Output vector is initialized with zeros. Expected values are 0, not None"""
        input = [3, 3, 3, 0.3]
        myNeuralNetwork = NeuralNetwork(input)

        myNeuralNetwork.init_output_vector()
        output_vector_tmp = myNeuralNetwork.get_output_vector()
        for ele in output_vector_tmp:
            self.assertEqual(ele, 0)
            self.assertNotEqual(ele, None)

if __name__ == '__main__':
    unittest.main()