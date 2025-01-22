import torch.nn as nn

"""
测试模型的FLOPs以及MACs
"""
class MyModel(nn.Module):
    def __init__(self, tracker):
        super(MyModel, self).__init__()
        # Your module definition here
        self.tracker = tracker

    def forward(self, img):
        # Your forward pass definition here
        return self.tracker.track(img)