"""
This will provide a upper level API to train a model.
"""

def train(model:nn.Module,dataloader:torch.data.Dataloader,epochs,validate=False):
    """
    Functions will take model as a input and should be agnostic of the type of the detector model,
    A custom dataset loader, and the number of epochs to train.
    """
    pass