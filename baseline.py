import numpy as np
import cv2 as cv2
import csv
import sys
import random
import collections
import matplotlib.pyplot as plt

#''' git commit -a -m "comment" '''
''' 
TO COMMIT ON GITHUB:

git add file_name --> to add file to a commit

git commit -a -m "comment"
-a --> to commit all files
git push origin master
'''


'''Set this to true to use a truncated version of the data of size smallDataSetSize'''
smallDataSet = True
smallDataSetSize = 100 #number of data points in the smaller data set (for both the test and train data)

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
    testing_data2 = None


    #using a smaller data set for testing
    if smallDataSet:
        trainCounter = 0
        testCounter = 0
        for line in dataLines:

            if trainCounter >= smallDataSetSize and testCounter >= smallDataSetSize: break
            
            inputList = None
            if line[2] == 'Training':
                
                if trainCounter >= smallDataSetSize: continue
                inputList = training_data
                trainCounter += 1
            
            elif line[2] == 'PublicTest' or line[2] == 'PrivateTest':
                
                if testCounter >= smallDataSetSize: continue
                inputList = testing_data1
                testCounter +=1

            #append data point
            if inputList != None: 
                pixels = [int(val) for val in line[1].split(" ")]
                inputList.append((pixels, int(line[0])))


    #using all the data
    else: 
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
        print evaluatePredictor(testExamples, predictor)

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
    # """
    # Implements d1 += scale * d2 for sparse vectors.
    # @param dict d1: the feature vector which is mutated.
    # @param float scale
    # @param dict d2: a feature vector.
    # """
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
        if (runningMin == None) or ((distance < runningMin)):
            runningMin = distance
            closetCentroidIndex = i

        if((distance == runningMin)): closetCentroidIndex = random.choice([closetCentroidIndex, i])
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
        -cluster assignments (clusters[i] = cluster data[i] is assigned to)
        -final centroids
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
        print "iteration: ", iteration
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
        iteration += 1

        #if iteration%10 == 0: print "iteration: ", iteration


    if(iteration != maxIterations): print "Converged!"
    
    '''print "---final centroids---"
    print "centroids: ", centroids
    print "iteration: ", iteration
    print "clusters: ", clusters
    '''

    return clusters, centroidsNew


def evaluateClusters(clusterAssignments, data, k):

    #groupMapping[guessed group number] --> True Group number
    kmeansGroup_to_trueGroup = detmineGroupMapping(clusterAssignments, data, k)
    correct = 0
    nDataPoints = len(clusterAssignments)
    for i in range(0, nDataPoints):
        if kmeansGroup_to_trueGroup[clusterAssignments[i]] == data[i][1]: correct +=1

    
    accuracy = float(correct)/float(nDataPoints)

    print "----RESULTS----"
    print "nDataPoints: ", nDataPoints
    print "k: ", k
    print "accuracy: ", accuracy



def detmineGroupMapping(clusterAssignments, data, k):
    '''
    Returns a dictionary with:
        -key: cluster number given by running kmeans (integer in range [0,k-1])
        -value: the input data's original cluster number for this grouping

    ex: kmeansGroup_to_trueGroup[guessed group number] --> True Group number
    '''

    #trueEmotionCounter[true Group number] = total number of this emotion type in data
    trueEmotionCounter = collections.Counter()
    clusterList = [] #list of length k, clusterList[0] = a counter of how many of each true emotion type were clustered to group 0
    for i in xrange(0, k): clusterList.append(collections.Counter())
    for i in range(0, len(clusterAssignments)):
        guessedGroup = clusterAssignments[i]
        trueGroup = data[i][1]
        trueEmotionCounter[trueGroup] += 1
        clusterList[guessedGroup][trueGroup] += 1

    clusterPercentList = []##list of length k, clusterPercentList[guessedGroup] = dictionary (key: true emotion type --> value: percent of this emotion type mapped to this guessed group)
    for i in xrange(0, k): clusterPercentList.append(dict())
    for guessedGroup in range(0, k):
        counter = clusterList[guessedGroup]
        for trueGroup, count in counter.items():  clusterPercentList[guessedGroup][trueGroup] = float(count)/float(trueEmotionCounter[trueGroup])


    print "cluster list: ", clusterList
    print "trueEmotionCounter: ", trueEmotionCounter
    print "clusterPercentList: ", clusterPercentList

    
    kmeansGroup_to_trueGroup = dict()
    for i in range(0,k):
        if len(clusterList[i].most_common(1)) == 0: kmeansGroup_to_trueGroup[i] = None
        else: kmeansGroup_to_trueGroup[i] = (clusterList[i].most_common(1))[0][0]
        #kmeansGroup_to_trueGroup[guessed group number] --> True Group number

    print "kmeansGroup --> trueGroup: ", kmeansGroup_to_trueGroup
    return kmeansGroup_to_trueGroup



    '''
    code to map a true group to the guessed group containing the highest percent of the true groups data points
    STILL DEBUGGING

    #determine mapping by assigning a true group to the guessed group containing the highest percent of the true groups data points
    kmeansGroup_to_trueGroup = dict()

    #assign the least prevelant emotion category to a guessed group first
    sortedTrueEmotionsReverseOrder = sorted(list(trueEmotionCounter))
    sortedTrueEmotions = []
    for item in reversed(sortedTrueEmotionsReverseOrder):
        sortedTrueEmotions.append(item)

    #sortedTrueEmotions = sorted(trueEmotionCounter.keys())
    print "sortedTrueEmotions: ", sortedTrueEmotions
    for trueGroup, count in sortedTrueEmotions:
        highestPercent = None
        bestGroup = None
        for guessedGroup, percentCounter in clusterPercentList: 
            if guessedGroup in kmeansGroup_to_trueGroup: continue
            if(bestGroup == None or percentCounter[trueGroup] > highestPercent):
                bestGroup = guessedGroup
                highestPercent = percentCounter[trueGroup]
        kmeansGroup_to_trueGroup[bestGroup] = trueGroup

    print "kmeansGroup --> trueGroup: ", kmeansGroup_to_trueGroup
    return kmeansGroup_to_trueGroup

    '''

def convertDataPointsToDictionaries(data):
    ''' 
    returns list of size len(data). 
    For the returned list, list[0] = dictionary representation of data[0] where 
    dictionary[pixelIndex] = pixelValue
    '''

    examples = [] #list of dictionarys (pixel index --> pixel value)
    for i in range(0, len(data)):
        pixels = data[i][0]
        pixelDict = {}
        for j in range(0, len(pixels)): pixelDict[j] = pixels[j]
        examples[i] = pixelDict

    return examples

def clusterData(data, centroids):
    '''
    This function assigns each data point in data, to a centroid in centroids
    return value:
        -list of cluster assignments where list[0] = cluster data[0] was assigned to
    '''

    nDataPoints = len(data)
    clusters = [-1]*nDataPoints
    for i in range(0, nDataPoints):
        dataPoint = data[i]
        closetCentroidIndex = returnClosetCentroid(dataPoint, centroids)
        clusters[i] = closetCentroidIndex

    return clusters


def runSurf(training_data, testing_data1, testing_data2):
    print "starting surf"
    pixelList = [pixels for pixels, emotion in training_data]

    
    row = []
    surfFeaturesList = []
    #for x in range(len(pixelList)):
    for x in range(1): 
        twoDArray = []
        for i in range(0, len(pixelList[x])):
            if i % 48 == 0 and i!= 0:
                twoDArray.append(row)
                row = []
            row.append(pixelList[x][i])
        twoDArray.append(row)
        print len(twoDArray)

        #surf = cv2.SURF(400)   
        sift = cv2.SIFT()
        #spoints = surf.detectAndCompute(np.uint8(np.array(twoDArray)), None)
        spoints = sift.detectAndCompute(np.uint8(np.array(twoDArray)), None)
        
        img2 = cv2.drawKeypoints(np.uint8(np.array(twoDArray)),spoints[0],None,(255,0,0),4)
        plt.imshow(img2),plt.show()
        #print spoints[1]
        surfFeaturesList.append(spoints)
    k = 7
    maxIter = 10
    
    #clusters = cv2.kmeans(np.array(surfFeaturesList), k, (cv2.TERM_CRITERIA_MAX_ITER, 10, .1), 1, cv2.KMEANS_RANDOM_CENTERS)
    #clusters = kmeans(surfFeaturesList, k, maxIter)
    #evaluateClusters(clusters, training_data, k)


def runBaselinePredictor(training_data, testing_data1, testing_data2):
    '''
    This function holds code to either:
    1. run stochastic gradient descent 
    2. run kmeans

    ** Note: if smallDataSet = true, the value of testing_data2 will be None **

    '''

    #set which test data set you want to test the baseline predictor on
    #note, all functions current set to use training_data for training
    testData = testing_data1

    '''stochastic gradient descent'''
    #learnPredictor(training_data, testing_data1, testing_data2, pixelIndexFeatureExtractor)


    '''some functions might want data points represented as dictionaries'''
    #dataAsDictionaries = convertDataPointsToDictionaries(training_data):


    '''Code to convert data from list of tuples (pixels, emotion) to just list of pixels 
    (required for kmeans) '''

    trainingPixelList = [pixels for pixels, emotion in training_data]
    testingPixelList = [pixels for pixels, emotion in testData]
    
    '''kmeans clustering'''

    k = 6
    maxIter = 30
    clusters, centroids = kmeans(trainingPixelList, k, maxIter)
    evaluateClusters(clusters, training_data, k)

    '''use centroids to cluster test data'''
    clusters = clusterData(testingPixelList, centroids)
    evaluateClusters(clusters, testData, k)


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
    runSurf(training_data, testing_data1, testing_data2)

if __name__ == '__main__':
  main()
