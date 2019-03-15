import tkinter as tk
import numpy
from neuronalnets import NeuralNetwork

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

HORIZONTAL_OFFSET = 300
INITIAL_HORIZONTAL_OFFSET = 100
VERTICAL_OFFSET = 100
INITIAL_VERTICAL_OFFSET = 100
RADIUS = 20

class NeuralNetworkDrawer:
    def __init__(self, NeuralNetwork):
        self.input_nodes_circle = NeuralNetwork.get_input_vector()
        self.input_node_circle_params = [self.input_nodes_circle, 0*HORIZONTAL_OFFSET, VERTICAL_OFFSET,RADIUS,'blue']
        self.hidden_nodes_circle = NeuralNetwork.get_hidden_vector()
        self.hidden_node_circle_params = [self.hidden_nodes_circle, 1*HORIZONTAL_OFFSET, VERTICAL_OFFSET,RADIUS,'red']
        self.output_nodes_circle = NeuralNetwork.get_output_vector()
        self.output_node_circle_params = [self.output_nodes_circle, 2*HORIZONTAL_OFFSET, VERTICAL_OFFSET,RADIUS,'green']
        self.create_canvas()
        pass

    def create_circle(self, x, y, r, **kwargs):
        x = x + INITIAL_HORIZONTAL_OFFSET
        y = y + INITIAL_VERTICAL_OFFSET
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return self.canvas.create_oval(x0, y0, x1, y1, **kwargs)

    def create_canvas(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, borderwidth=2, highlightthickness=0, bg="white")
        self.canvas.grid()

    @staticmethod
    def calculate_circle_param(input_list):
        output_list = {}
        for idx, ele in enumerate(input_list[0]):
            output_list[idx] = [input_list[1], idx*input_list[2], input_list[3], float(ele), input_list[4]]

        print(output_list)
        return output_list

    def update(self):
        input_node = NeuralNetworkDrawer.calculate_circle_param(self.input_node_circle_params)
        hidden_node = NeuralNetworkDrawer.calculate_circle_param(self.hidden_node_circle_params)
        output_node = NeuralNetworkDrawer.calculate_circle_param(self.output_node_circle_params)

        for values in input_node.values():
            self.create_circle(values[0],values[1], values[2], fill=values[4])

        for values in hidden_node.values():
            self.create_circle(values[0],values[1], values[2], fill=values[4])

        for values in output_node.values():
            self.create_circle(values[0], values[1], values[2], fill=values[4])

        self.root.mainloop()

def create_neural_net():
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

    return myNeuralNetwork

if __name__ == "__main__":
    myNetwork = create_neural_net()

    myNetworkDrawer = NeuralNetworkDrawer(myNetwork)

    myNetworkDrawer.update()
