import numpy as np
from tqdm import tqdm

def classcount(loader):
    n_train = len(loader)

    class_weight = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    with tqdm(total=n_train, desc='Class Count Assessment', unit='batch', disable = False, leave=True) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            (unique, counts) = np.unique(true_masks, return_counts=True)
            frequencies = np.asarray((unique, counts))
            # print(frequencies.shape)
            for i in range(frequencies.shape[1]):
                class_weight[frequencies[0,i]] += frequencies[1,i]
            pbar.update()

    # print(class_weight)
    class_weight = class_weight[:-1].min()/class_weight
    class_weight[-1]=0

    return class_weight
