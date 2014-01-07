import json
from pprint import pprint
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from random import randint

json_file = open('sta16-train.json')
json_data = json.load(json_file)
json_file.close()

num_inputs = 16*16  # 16 by 16 pixels
num_outputs = 10    # 10 digits of output, 0-9, although 0 is marked as 10, inexplicably
num_hidden_nodes = 64
training_epochs = 100

dataset = SupervisedDataSet(num_inputs,num_outputs)

for sample in json_data:
    dataset.addSample(sample['input'], sample['target'])

net = buildNetwork(num_inputs,
        num_hidden_nodes,
        num_outputs,
        outclass=SoftmaxLayer)
trainer = BackpropTrainer(net,dataset)

for i in range(1,training_epochs+1):
    print("{0} of {1} epochs complete".format(i, training_epochs))
    trainer.train()

json_file = open('sta16-test.json')
json_data = json.load(json_file)
json_file.close()

for idx, test_item in enumerate(json_data):
    result = net.activate(test_item['input'])
    print "Item({0}): target was {1}, output was {2}".format(idx, test_item['target'], result)

