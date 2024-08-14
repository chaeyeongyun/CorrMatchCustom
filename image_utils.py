import numpy as np
import matplotlib.pyplot as plt 
import os
   
def target_to_colormap(target:np.ndarray, colormap:np.ndarray):
    show = colormap[target] # (N, H, W, 3)
    return show
 
def save_img(img_dir:str, filename:str, img:np.ndarray):
    plt.imsave(os.path.join(img_dir, filename), img)

def batch_to_grid(array:np.ndarray, transpose=True):
    array = array.transpose(0, 2, 3, 1) if transpose else array
    cat_img = np.squeeze(np.concatenate(np.split(array, len(array), axis=0), axis=1), axis=0)
    return cat_img

def pred_to_detailed_colormap(pred:np.ndarray, target:np.ndarray, colormap:np.ndarray):
    labels = np.unique(target).tolist()
    num_classes = len(labels)
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    
    for label in labels:
        pred_label[(pred_label == label) & (target != label)] = label + num_classes
    # crop TP:2,red FP 5,yellow
    # weed TP:1,blue FP 4,orange
    # BG TP:0,black FP 3, gray
    # https://www.color-hex.com/color/e69138
    if num_classes == 3:
        colormap = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0.5, 0.5, 0.5], [230/255, 145/255, 56/255], [1, 217/255, 102/255]]) # (3,3)->(6,3)
    else:
        raise NotImplementedError
    show = colormap[pred_label] # (N, H, W, 3)
    return show

def make_test_detailed_img(input, pred, target, colormap:np.ndarray=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
    input = batch_to_grid(input.transpose(0, 3, 1, 2))
    pred = batch_to_grid(pred_to_detailed_colormap(pred, target, colormap=colormap), transpose=False)
    target = batch_to_grid(target_to_colormap(target, colormap=colormap), transpose=False)
    viz_v1 = np.concatenate((input, target, pred), axis=1)
    return viz_v1
    