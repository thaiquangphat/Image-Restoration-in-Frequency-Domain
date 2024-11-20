import numpy as np

def gaussianKernel(M, N, D0):
    H = np.zeros((M, N), dtype=np.float_)
    centerX, centerY = M/2, N/2

    for u in range(M):
        for v in range(N):
            dist = (u - centerX)**2 + (v - centerY)**2
            H[u, v] = np.exp(-dist/(2*D0**2))

    return H

def butterworthKernel(M, N, D0):
    H = np.zeros((M, N), dtype=np.float_)
    centerX, centerY, n = M/2, N/2, 2.25

    for u in range(M):
        for v in range(N):
            dist = np.sqrt((u - centerX)**2 + (v - centerY)**2)
            if dist == 0:
                H[u, v] = 1
            else:
                H[u, v] = 1 / (1 + (dist/D0)**(2*n))
    
    return H