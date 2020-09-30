
def parse_model_config(path):
    """
    Parses the configuration (cfg) file and returns a list containing dictionary
    with modules defination.
    """
    with open(path , 'r') as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
        module_defs = []
        for line in lines:
            if line.startswith('['): # This marks the start of a new block
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()

        return module_defs

def parse_path(path):
    """
    Parse cfg path to return model name.
    Args:
        path(str):
    """
    if (path.endswith('.cfg')):
        n = path.split('/')[-1]
        n = n.split('.')[0]
        return n
    else:
        raise Exception(f'Broken path to cfg file.')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
    #we will use biggest k, and calculate all precisions from 0 to k
        maxk = max(topk)
        batch_size = target.size(0)
    
        _, pred = output.topk(input=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
   
        correct = pred.eq(target.view(1, -1).expand_as(pred))
   
        res = []
   
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res