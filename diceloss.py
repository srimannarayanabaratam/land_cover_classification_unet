import torch

def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    # with torch.no_grad():
    #   y_pred = torch.argmax(y_pred, dim=1).float()
    # y_pred.requires_grad=True

    y_pred = torch.softmax(y_pred, dim=1)
    # print("Prediction", y_pred.shape)
    # print("Truth", y_true.shape)

    # y_true_f = torch.flatten(torch.nn.functional.one_hot(y_true.to(torch.int64), num_classes=7)[...,1:])
    y_true_f = torch.flatten(torch.nn.functional.one_hot(y_true.to(torch.int64), num_classes=7))
    # y_pred_f = torch.flatten(torch.nn.functional.one_hot(y_pred.to(torch.int64),num_classes=7)[...,1:])

    # y_pred_f = torch.flatten(y_pred.permute(0,2,3,1)[...,1:])
    y_pred_f = torch.flatten(y_pred.permute(0,2,3,1))

    # print("Prediction_flat", y_pred_f.shape)
    # print("Truth_flat", y_true_f.shape)

    
    intersect = torch.sum(y_true_f * y_pred_f, axis=-1)

    denom = torch.sum(y_true_f + y_pred_f, axis=-1)
    return torch.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_9cat(y_true, y_pred)