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

def createKernel(image, size, x, y):
    M, N = image.shape
    arr = []
    r = size // 2

    for i in range(-r, r+1):
        for j in range(-r, r+1):
            X = x + i
            Y = y + j

            if 0 <= X < M and 0 <= Y < N:
                arr.append((X, Y))

    return arr

def contraharmonic(image, size, Q):
    restored = np.zeros_like(image, dtype='uint8')
    M, N = image.shape

    for x in range(M):
        for y in range(N):
            kernel = createKernel(image, size, x, y)
            
            sumQ1 = np.sum([image[u, v]**(Q+1) for (u, v) in kernel])
            sumQ = np.sum([image[u, v]**(Q) for (u, v) in kernel])

            if sumQ == 0:
                median = 0
            else:
                median = sumQ1 / sumQ

            restored[x, y] = median
    
    return restored.astype(np.uint8)