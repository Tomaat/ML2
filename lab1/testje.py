#%pylab inline
#pylab.rcParams['figure.figsize'] = (10, 6)
import numpy as np
# Signal generators
def sawtooth(x, period=0.2, amp=1.0, phase=0.):
    return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp

def sine_wave(x, period=0.2, amp=1.0, phase=0.):
    return np.sin((x / period - phase) * 2 * np.pi) * amp

def square_wave(x, period=0.2, amp=1.0, phase=0.):
    return ((np.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5) * 2 * amp

def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
    return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5) * 2 * amp

def random_nonsingular_matrix(d=2):
    """
    Generates a random nonsingular (invertible) matrix of shape d*d
    """
    epsilon = 0.1
    A = np.random.rand(d, d)
    while abs(np.linalg.det(A)) < epsilon:
        A = np.random.rand(d, d)
    return A

num_sources = 5
signal_length = 500
t = np.linspace(0, 1, signal_length)
S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25), np.random.randn(t.size)].T


def make_mixtures(S,A):
    X = np.dot(A,S)
    return X
	
def plot_histograms(X):
    figure()
    for i,row in enumerate(X):
        subplot(X.shape[0],1,i+1)
        hist(row)

def phi_0(a):
    return -np.tanh(a)
def p_0(a):
    return 1/np.cosh(a)
def phi_1(a):
    return -a+np.tanh(a)
def p_1(a):
    return np.exp(-a**2/2)*np.cosh(a)
def phi_2(a):
    return -a**3
def p_2(a):
    return np.exp(-a**4/4)
def phi_3(a):
    return -6*a/(a**2+5)
def p_3(a):
    return 1/(a**2+5)**3

phi = [phi_0, phi_1, phi_2, phi_3]
p = [p_0, p_1, p_2, p_3]

def whiten(X):
    U,s,V = np.linalg.svd(np.dot(X,X.T))
    return np.dot(U.T,X)