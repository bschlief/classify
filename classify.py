import json
from pprint import pprint
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from random import randint

json_file = open("data.json")
json_data = json.load(json_file)
json_file.close()

dataset = SupervisedDataSet(2,1)

for sample in json_data:
    input0 = sample['in'][0]/9.0
    input1 = sample['in'][1]/9.0
    output0 = sample['out'][0]/9.0
    dataset.addSample((input0,input1), (output0,))

net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
#net = buildNetwork(2, 10, 1, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net,dataset)

for i in range(1,100000):
    trainer.train()

for i in range(1,100):
    one = randint(0,9)
    two = randint(0,9)
    print "result of {0} + {1}".format(one/9.0, two/9.0)
    print net.activate((one,two))*9.0
