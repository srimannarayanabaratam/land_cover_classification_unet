import torch

def dice_coef_9cat(y_true, y_pred, smooth=1e-7):

    y_pred = torch.softmax(y_pred, dim=1)
    y_true_f = torch.flatten(torch.nn.functional.one_hot(y_true.to(torch.int64), num_classes=7))
    y_pred_f = torch.flatten(y_pred.permute(0,2,3,1))
    intersect = torch.sum(y_true_f * y_pred_f, axis=-1)
    denom = torch.sum(y_true_f + y_pred_f, axis=-1)
    return torch.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat_loss(y_true, y_pred):
    return 1 - dice_coef_9cat(y_true, y_pred)
