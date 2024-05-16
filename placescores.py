
## These values should all be compared for normalized place cells, with values between 0 and 1
import numpy as np
import cv2



## Smoothness
def smoothness(pc):
    smooth_pc = cv2.GaussianBlur(pc, (5,5), sigmaX=1, sigmaY=0)
    return np.mean(np.abs(pc - smooth_pc))


## Consistency
# eps is a small constant
def consistency(pc, eps):
    return np.mean(np.where(pc < eps, 1, 0)) + np.mean(np.where(pc > 1-eps, 1, 0))



# Sparsity
def sparsity(pc):
    lambda_bar = np.mean(pc.flatten())
    lambda_sqrd_bar = np.mean(pc.flatten()**2)
    
    return lambda_bar**2 / lambda_sqrd_bar

def pc_score(pc, eps):
    return (-100*smoothness(pc) + 10*consistency(pc, eps) - 10*sparsity(pc))