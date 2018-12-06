import os
from time import strftime, localtime



def prepare_model_dir(WorkDir):
    # Create results directory
    result_path = os.getcwd() + WorkDir + '/' + strftime('%b_%d_%H_%M_%S', localtime())
    os.mkdir(result_path)

    return result_path
