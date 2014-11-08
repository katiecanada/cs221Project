''' git commit -a -m "comment" '''

def HingeLossGradient(w, y, phi):
    '''
    Given weights (w), y, and phi, the function computes and returns the
    corresponding Hinge Loss Gradient. 
    '''
    check = dotProduct(w, phi) * y
    if check > 1: return {}
    else: 
        phi_new = {}
        increment(phi_new, -1 * y, phi)
        return phi_new


def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    def predictor(x):
        if(dotProduct(weights, featureExtractor(x)) < 0): return -1 
        else: return 1

    eta = 1
    numIters = 20
    for i in range(numIters):
        eta = 1 / math.sqrt(i + 1) #step size dependent on the interation number
        for x, y in trainExamples:
            phi = featureExtractor(x)
            HLG = HingeLossGradient(weights, y, phi)
            increment(weights, -1*eta, HLG)
        #print evaluatePredictor(trainExamples, predictor)
        #print evaluatePredictor(testExamples, predictor)

    # END_YOUR_CODE
    return weights

    def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

    