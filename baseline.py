'''
    TO COMMIT ON GITHUB:
    
    git add file_name --> to add file to a commit
    
    git commit -a -m "comment"
    -a --> to commit all files
    git push origin master
    '''

import csv
import sys
import random

def parseData(file_name):
    '''
        Function Purpose: parse input data from a csv file
        
        Return Value: 3-tuple: (training_data, testing_data1, testing_data2)
        - each element in training_data, testing_data1, and testing_data2 is a 2-tuple: (pixels, emotion category)
        - pixels: a list of pixel color values. The index of the color value represents the index of the pixel
        - emotion category: the emotion given in the image. this is a value in the range [0,6]
        
        
        '''
    
    data = open(file_name, 'r')
    dataLines = csv.reader(data)
    training_data = [] #list of tuples (pixels, emotion category) where pixels is a list ints representing the
    #color value for each pixel and 'emotion category' is an int is in the range [0, 6]
    testing_data1 = []
    testing_data2 = []
    for line in dataLines:
        #choose which list to append data point to
        inputList = None
        if line[2] == 'Training': inputList = training_data
        if line[2] == 'PublicTest': inputList = testing_data1
        elif line[2] == 'PrivateTest': inputList = testing_data2
        
        #append data point
        if inputList != None:
            pixels = [int(val) for val in line[1].split(" ")]
            inputList.append((pixels, int(line[0])))

return (training_data, testing_data1, testing_data2)


''' ----- STOCHATIC GRADIENT DESCENT CODE -------'''

def evaluatePredictor(examples, predictor):
    '''
        predictor: a function that takes an x and returns a predicted y.
        Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
        of misclassiied examples.
        '''
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

def pixelIndexFeatureExtractor(x):
    '''
        Feature Extractor Function
        
        input: list of pixel values (ints) that correspond to an image
        output: phi(x) represented as a dictionary
        -feature (index of the pixel) --> value (color value of the pixel)
        
        '''
    featureVector = dict()
    for i in range(len(x)):
        featureVector[i] = x[i]
    
    return featureVector


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

def learnPredictor(trainExamples, testExamples1, testExamples2, featureExtractor):
    '''
        Given |trainExamples| and |testExamples| (each one is a list of (x,y)
        pairs), a |featureExtractor| to apply to x, and the number of iterations to
        train |numIters|, return the weight vector (sparse feature vector) learned.
        
        You should implement stochastic gradient descent.
        
        Note: only use the trainExamples for training!
        You should call evaluatePredictor() on both trainExamples and testExamples
        to see how you're doing as you learn after each iteration.
        '''
    
    #each category has its own set of weights
    weightList = [{}]*6
    #weights = {}  # feature => weight
    
    def predictor(x):
        '''
            -returns argMax over i of: dotProduct(weightList[i], featureExtractor(x))
            -this argMax represents the category that gives the highest score for the given x
            '''
        maxScore = None
        bestCategory = None
        for i in range(len(weights)):
            score = dotProduct(weightList[i], featureExtractor(x))
            if maxScore == None or score > maxScore: bestCategory = i
        
        return bestCategory
    
    eta = 1
    numIters = 20
    for i in range(numIters):
        eta = 1 / ((i + 1)**(1/2.0)) #step size dependent on the interation number
        for x, y in trainExamples:
            phi = featureExtractor(x)
            HLG = HingeLossGradient(weights, y, phi)
            increment(weights, -1*eta, HLG)
        print evaluatePredictor(trainExamples, predictor)
        print evaluatePredictor(testExamples1, predictor)
        print evaluatePredictor(testExamples2, predictor)
    
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



''' ---------- KMEANS CLUSTERING CODE ----------- '''



def euclideanDist(point1, point2):
    '''
        ----
        Function Purpose: Given two points (inputted as lists), this function returns their Eucliedean
        distance
        
        Arguments:
        -two points represented as lists
        Return Value:
        -the Eculidean distance between the two points
        ----
        '''
    
    nDimensions = len(point1)
    sqSum = 0
    for i in range(nDimensions):
        sqDifference = (point1[i] - point2[i])**(2.0)
        sqSum += sqDifference
    
    return sqSum**(1.0/2.0)



def returnAverage(pointList):
    '''
        ----
        Function Purpose: Given a list of points, this function returns their average
        Arguments:
        -a list of points
        Return Value:
        -the average of the points in the list
        ----
        '''
    nPoints = len(pointList)
    if nPoints == 0: return None
    
    nDimensions = len(pointList[0])
    average = [0] * nDimensions
    
    for point in pointList:
        for i in range(0, nDimensions):
            average[i] += 1.0/nPoints * point[i]
    return average

def returnClosetCentroid(point, centroidsPrev):
    '''
        ----
        Function Purpose: Given a point and list of centroids, this function returns the index of the centroid
        closest to the point. Note: if multiples centroids are equidistant to the point, the
        function randomly selects one of the equidistant centroids to return
        
        Arguments:
        -a list representing a specfic point and a list of centroids
        Return Value:
        -the index of the closest centroid to the point
        ----
        '''
    runningMin = None
    closetCentroidIndex = 0 #default to being the centroid at index 0
    nCentroids = len(centroidsPrev)
    
    for i in range(nCentroids):
        centroid = centroidsPrev[i]
        distance = euclideanDist(centroid, point)
        if (runningMin == None) or (distance < runningMin):
            runningMin = distance
            closetCentroidIndex = i
        if(distance == runningMin): closetCentroidIndex = random.choice([closetCentroidIndex, i])
    return closetCentroidIndex

def notConverged(centroidsPrev, centroidsNew):
    '''
        ----
        Function Purpose: Given the new centroids and the centroids calculated in the last iteration of the
        algorithm, this function returns whether or not the two centroid lists are equivalent
        
        Arguments:
        -two lists of centroids
        Return Value:
        -True: the two lists are different
        -False: the two lists are the same
        ----
        '''
    count = len([centroid for centroid in centroidsNew if centroid not in centroidsPrev])
    count += len([centroid for centroid in centroidsPrev if centroid not in centroidsNew])
    
    return count != 0


def kmeans(data, k, maxIterations):
    '''
        ----
        Function Purpose: This function performs the kmeans clustering algorithm. The
        algorithm stops iterating once the calculated centroids converge or the max
        number of iterations is achieved.
        
        Arguments:
        -data, a list of centroids, k, max number of iterations
        Return Value:
        -cluster assignements (clusters[i] = cluster data[i] is assigned to)
        ----
        '''
    
    print "starting kmean clustering"
    
    centroidsPrev = [[]] * k
    nDataPoints = len(data)
    clusters = [-1] *  nDataPoints
    centroidsNew = [{}] * k
    
    #initialize centroids to a random sample of size K from examples
    for i in range(k):
        centroidsNew[i] = random.choice(data)
    
    
    iteration = 0
    while(notConverged(centroidsPrev, centroidsNew) and iteration < maxIterations):
        print "starting iteration: ", iteration
        centroidsPrev = list(centroidsNew)
        
        #assign points to a centroid
        for i in range(0, nDataPoints):
            point = data[i]
            closetCentroidIndex = returnClosetCentroid(point, centroidsPrev)
            clusters[i] = closetCentroidIndex
        
        #recalculate centroids
        for centroidIndex in range(0, k):
            points = [data[i] for i in range(nDataPoints) if clusters[i]==centroidIndex]
            average = returnAverage(points)
            if average != None: centroidsNew[centroidIndex] = average
        
        print "finished iteration: ", iteration
        iteration += 1
    
    #if iteration%10 == 0: print "iteration: ", iteration
    
    centroids = centroidsNew

if(iteration != maxIterations) print "Converged!"
    
    '''print "---final centroids---"
        print "centroids: ", centroids
        print "iteration: ", iteration
        print "clusters: ", clusters
        '''
    
    return clusters


def evaluateClusters(clusterAssignments, data, k):
    correct = 0
    nDataPoints = len(clusterAssignments)
    for i in range(0, nDataPoints):
        if clusterAssignments[i] == data[i][1]: correct +=1
    
    
    accuracy = float(correct)/float(nDataPoints)

print "----RESULTS----"
    print "nDataPoints: ", nDataPoints
    print "k: ", k
    print "accuracy: ", accuracy


def runBaselinePredictor(training_data, testing_data1, testing_data2):
    #learnPredictor(training_data, testing_data1, testing_data2, pixelIndexFeatureExtractor)
    
    '''
        code to convert data in a data list into a list of dictionaries (key: pixel index --> value: pixel color value)
        
        
        examples = [] #list of dictionarys (pixel index --> pixel value)
        for i in range(0, len(training_data)):
        pixels = training_data[i][0]
        pixelDict = {}
        for j in range(0 len(pixels)): pixelDict[j] = pixels[j]
        examples[i] = pixelDict
        
        '''
    
    
    '''
        Code to convert data from list of tuples (pixels, emotion) to just list of pixels
        
        '''
    pixelList = [pixels for pixels, emotion in testing_data2]
    
    k = 7
    maxIter = 10
    clusters = kmeans(pixelList, k, maxIter)
    evaluateClusters(clusters, training_data, k)


def testInputData(training_data, testing_data1, testing_data2):
    print "---- TRAINING DATA ----"
    print "# data points: ", len(training_data)
    print "1st data point: "
    print "Emotion: ", training_data[0][1]
    print "Pixels: ", training_data[0][0]
    print "---- TESTING DATA1 ----"
    print "# data points: ", len(testing_data1)
    print "1st data point: "
    print "Emotion: ", testing_data1[0][1]
    print "Pixels: ", testing_data1[0][0]
    print "---- TESTING DATA2 ----"
    print "# data points: ", len(testing_data2)
    print "1st data point: "
    print "Emotion: ", testing_data2[0][1]
    print "Pixels: ", testing_data2[0][0]
    print "---- TESTING DATA2 ----"
    print "# data points: ", len(testing_data2)
    print "last data point: "
    print "Emotion: ", testing_data2[-1][1]
    print "Pixels: ", testing_data2[-1][0]




def main():
    if len(sys.argv) < 2: raise Exception("no input file given")
    training_data, testing_data1, testing_data2 = parseData(sys.argv[1])
    #testInputData(training_data, testing_data1, testing_data2)
    
    runBaselinePredictor(training_data, testing_data1, testing_data2)


if __name__ == '__main__':
    main()
