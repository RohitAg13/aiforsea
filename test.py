
import argparse
from os import path

from fastai.vision import *
from fastai.metrics import error_rate

import pandas as pd
import pickle

def export(test_folder):
    if path.exists('data/idx2classes.pickle'):
        with open('data/idx2classes.pickle','rb') as f:
            classes = pickle.load(f)
    else:
        print('Classes file not present')
        return
         
    path = Path('data/')
    if path.exists('data/exported-model.pkl'):
        print('loading model')
        learn = load_learner(path,'exported-model.pkl',test= ImageList.from_folder(test_folder))
    else:
        print('model file not found in data/')
        return

    print('starting prediction model')
    preds = learn.get_preds(ds_type=DatasetType.Test)[0]
    print('prediction complete')
    print('exporting')
    pred_confidence = [pred[np.argmax(pred.numpy())] for pred in preds]

    pred_class = [classes[(np.argmax(pred.numpy()))] for pred in preds]

    pred_category = [np.argmax(pred.numpy()) for pred in preds]

    result = pd.DataFrame()

    result['filename'] = learn.data.test_ds.items
    result['confidence'] = pred_confidence
    result['class_name'] = pred_class
    result['class_category'] = pred_category

    result.to_csv('result.csv',index=False)

    print('export complete to result.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="test folder name")
    args = parser.parse_args()
    export(args.test)

