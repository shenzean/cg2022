import numpy as np
from scipy import interpolate

def sampleCubicSplinesWithDerivative(points, tangents, resolution):

    resolution = float(resolution)
    points = np.asarray(points)
    nPoints, dim = points.shape

    # Parametrization parameter s.
    dp = np.diff(points, axis=0)                 # difference between points
    dp = np.linalg.norm(dp, axis=1)              # distance between points
    d = np.cumsum(dp)                            # cumsum along the segments
    d = np.hstack([[0],d])                       # add distance from first point
    l = d[-1]                                    # length of point sequence
    nSamples = int(l/resolution)                 # number of samples
    s,r = np.linspace(0,l,nSamples,retstep=True) # sample parameter and step

    # Bring points and (optional) tangent information into correct format.
    assert(len(points) == len(tangents))
    data = np.empty([nPoints, dim], dtype=object)
    for i,p in enumerate(points):
        t = tangents[i]
        # Either tangent is None or has the same
        # number of dimensions as the point p.
        assert(t is None or len(t)==dim)
        fuse = list(zip(p,t) if t is not None else zip(p,))
        data[i,:] = fuse

    # Compute splines per dimension separately.
    samples = np.zeros([nSamples, dim])
    for i in range(dim):
        poly = interpolate.BPoly.from_derivatives(d, data[:,i])
        samples[:,i] = poly(s)
    return samples
    # Input.
points = []
tangents = []
resolution = 0.2
points.append([0.,0.]); tangents.append([1,1])
points.append([3.,4.]); tangents.append([1,0])
points.append([5.,2.]); tangents.append([0,-1])
points.append([3.,0.]); tangents.append([-1,-1])
points = np.asarray(points)
tangents = np.asarray(tangents)

# Interpolate with different tangent lengths, but equal direction.
scale = 1.
tangents1 = np.dot(tangents, scale*np.eye(2))
samples1 = sampleCubicSplinesWithDerivative(points, tangents1, resolution)
scale = 2.
tangents2 = np.dot(tangents, scale*np.eye(2))
samples2 = sampleCubicSplinesWithDerivative(points, tangents2, resolution)
scale = 0.1
tangents3 = np.dot(tangents, scale*np.eye(2))
samples3 = sampleCubicSplinesWithDerivative(points, tangents3, resolution)

# Plot.
import matplotlib.pyplot as plt
plt.scatter(samples1[:,0], samples1[:,1], marker='o', label='samples1')
plt.scatter(samples2[:,0], samples2[:,1], marker='o', label='samples2')
plt.scatter(samples3[:,0], samples3[:,1], marker='o', label='samples3')
plt.scatter(points[:,0], points[:,1], s=100, c='k', label='input')
plt.axis('equal')
plt.title('Interpolation')
plt.legend()
plt.show()