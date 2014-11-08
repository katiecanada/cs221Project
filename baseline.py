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