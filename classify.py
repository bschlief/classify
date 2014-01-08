import json
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

#Constants
num_inputs = 16*16  # 16 by 16 pixels
num_outputs = 10    # 10 digits of output, 0-9
num_hidden_nodes = 64
max_training_epochs = 200
verbose = False
reuse_last_network = True


# Convert target integer to a vector with a 1 in the
# appropriate index.
# for example:
#   target = 8 means that index 8 is 1, otherwise vector is 0s
#         0  1  2  3  4  5  6  7  8  9
#   8 => [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
def int_to_vector(inpt, length):
    vec = [0 for x in range(length)]
    vec[inpt] = 1
    return vec

dataset = SupervisedDataSet(num_inputs, num_outputs)
with open('sta16-train.json', 'r') as f:
    train_json_data = json.load(f)

    for sample in train_json_data:
        target_vector = int_to_vector(int(sample['target']), num_outputs)
        dataset.addSample(sample['input'], target_vector)

if (reuse_last_network):
    print "Reusing last network..."
    net = NetworkReader.readFrom('network.xml')
else:
    print "Creating new network..."
    net = buildNetwork(num_inputs, num_hidden_nodes, num_outputs, bias=True, outclass=SoftmaxLayer)

trainer = BackpropTrainer(net, dataset)
results = trainer.trainUntilConvergence(verbose = True, maxEpochs=max_training_epochs)
NetworkWriter.writeToFile(net, 'network.xml')

print "{0} epochs consumed training".format(len(results[0])-1)

with open('sta16-test.json', 'r') as f:
    test_json_data = json.load(f)

    num_correct = 0
    for idx, test_item in enumerate(test_json_data):
        result = net.activate(test_item['input'])
        guess = result.argmax()
        if (test_item['target'] == guess):
            num_correct = num_correct + 1
        if (verbose):
            print "Item({0}): target was {1}, guess was {2}, output was {3}".format(idx, test_item['target'], guess, result)
    print "{0} out of {1} correct".format(num_correct, len(test_json_data))



