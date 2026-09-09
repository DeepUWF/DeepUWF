import numpy as np
import torch


def model_update(model_local, model_personal, alpha):
    '''
    Algorithms from APFL
    model_local, model_personal: model.modules
    '''
    for l_params, p_params in zip(model_local.parameters(), model_personal.parameters()):
        p_params.data = alpha * p_params.data + (1-alpha)*l_params.data
        


def alpha_update(model_local, model_personal, alpha, eta):
    '''
    Algorithms from APFL
    model_local, model_personal: model.modules
    '''
    grad_alpha = 0
    for l_params, p_params in zip(model_local.parameters(), model_personal.parameters()):
        dif = p_params.data - l_params.data
        grad = alpha * p_params.grad.data + (1-alpha)*l_params.grad.data
        grad_alpha += dif.view(-1).T.dot(grad.view(-1))
    grad_alpha += 0.02 * alpha

    alpha_n = alpha - eta*grad_alpha
    alpha_n = np.clip(alpha_n.item(),0.0,1.0)
    
    return alpha_n