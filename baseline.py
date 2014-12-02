import numpy as np
#import cv2 as cv2
import csv
import sys
import random
import collections
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy.cluster.vq


'''Set this to true to use a truncated version of the data of size smallDataSetSize'''
smallDataSet = True
smallDataSetSize = 500 #number of data points in the smaller data set (for both the test and train data)


'''
Note: Entire Data set contains
-Angry: 4953 data points
-Disgust: 547 data points
-Fear: 5121 data points
-Happy: 8989 data points
-Sad: 6077 data points
-Surprised: 4002 data points
-Neutral: 6198 data points

Totaling to: 35,887 data points
'''
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

    emotionCounter = [0]*7

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
                emotion = int(line[0])
                emotionCounter[emotion] += 1
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
                emotion = int(line[0])
                emotionCounter[emotion] += 1
                inputList.append((pixels, int(line[0])))

    print "emotionCounter: ", emotionCounter
    return (training_data, testing_data1, testing_data2)


''' ----- STOCHATIC GRADIENT DESCENT CODE -------'''

def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    i = 0
    for x, y in examples:
        #if i % 25 == 0: print "actual: ", y, "; predicted: ", predictor(x)
        i += 1
        if predictor(x) != y:
            error += 1
    return 1.0 * error / float(len(examples))

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


def getMaxBranchIndex(weightList, y, phi):
    '''determines and returns the arg max over i of: 
    {dotProduct(weightList[i], phi) - dotproduct(weightList[y], phi) + 1 * (indicator i equals not y)}'''

    branchValues = []
    true_weights = weightList[y]

    for i in range(0, len(weightList)):
        d = dotProduct(weightList[i], phi) - dotProduct(true_weights, phi)
        if y != i: d += 1
        branchValues.append(d)

    maxValue = None
    maxBranchIndex = None

    for i in range(0, len(branchValues)):
        if maxValue == None or branchValues[i] > maxValue:
            maxValue = branchValues[i]
            maxBranchIndex = i

    return maxBranchIndex


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
        # weights[0] = dictionary: key(feature) => value(weight)
    weightList = [{}, {}, {}, {}, {}, {}, {}]

    def predictor(x):
        '''
        -returns argMax over i of: dotProduct(weightList[i], featureExtractor(x))
        -this argMax represents the category that gives the highest score for the given x
        '''
        maxScore = None
        bestCategory = None
        phi = featureExtractor(x)
        for i in range(0, len(weightList)):
            score = dotProduct(weightList[i], phi)
            if maxScore == None or score > maxScore: 
                bestCategory = i
                maxScore = score

        return bestCategory

    eta = 1
    numIters = 30
    for i in range(numIters):
        eta = 1 / ((i + 1)**(1/2.0)) #step size dependent on the interation number
        for x, y in trainExamples:
            phi = featureExtractor(x)

            dominantWeightIndex = getMaxBranchIndex(weightList, y, phi)
            if dominantWeightIndex != y:
                HLG = phi
                incrementWeightList(weightList, y, eta, phi)
                incrementWeightList(weightList,dominantWeightIndex,-1*eta, HLG)


        print "--- iteration: ", i, " ----"
        print "train error: ", evaluatePredictor(trainExamples, predictor)
        print "test error: ", evaluatePredictor(testExamples, predictor)

    return weightList

def incrementWeightList(weightList, index, scale, d2):
    # """
    # Implements weightList[index] += scale * d2 for sparse vectors.
    # @param weightList[index]: the feature vector which is mutated.
    # @param float scale
    # @param dict d2: a feature vector.
    # """
    for f, v in d2.items():
        weightList[index][f] = weightList[index].get(f, 0) + v * scale

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
    if point2 != []:
        for i in range(nDimensions):
            sqDifference = (point1[i] - point2[i])**(2.0)
            sqSum += sqDifference
    else:
        for i in range(nDimensions):
            sqDifference = (point1[i])**(2.0)
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
        for i in range(0, len(point)):
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
    # count = 0
    # for centroid in centroidsNew:
    #     for point in centroid:
    #         print point
    #         if point not in centroidsPrev.any():
    #             count+=1

    count += len([centroid for centroid in centroidsPrev if centroid not in centroidsNew])

    return count != 0


def kmeans(data, k, maxIterations):
    '''
    ----
    Function Purpose: This function performs the kmeans clustering algorithm. The
    algorithm stops iterating once the calculated centroids converge or the max 
    number of iterations is achieved.

    Arguments: 
        -data
        -k
        -max number of iterations
    Return Value:
        -cluster assignments (clusters[i] = cluster data[i] is assigned to)
        -final centroids
    ----
    '''

    print "starting kmean clustering"

    centroidsPrev = [] 
    for i in range(0, k):
        centroidsPrev.append([])

    nDataPoints = len(data)
    
    clusters = []
    for i in range(0, nDataPoints): clusters.append(-1)
    centroidsNew = []
    for i in range(0, k):
        centroidsNew.append([])

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
        if clusterAssignments[i] not in kmeansGroup_to_trueGroup: continue
        if kmeansGroup_to_trueGroup[clusterAssignments[i]] == data[i][1]: correct +=1

    
    accuracy = float(correct)/float(nDataPoints)

    print "----RESULTS----"
    print "nDataPoints: ", nDataPoints
    print "k: ", k
    print "accuracy: ", accuracy



def detmineGroupMapping(clusterAssignments, data, k):
    '''
    Returns a dictionary with:
        -key: cluster index (integer in range [0,k-1])
        -value: emotion category assigned to this cluster (integer in range[0,k-1])

    ex: kmeansClusterIndex_to_emotion[clusterIndex] --> emotion

    This function also provides code for two different approaches for cluster/emotion assignemnt
    '''



    #emotionCounter[emotion] = number of data points in the data set classified as this emotion
    emotionCounter = collections.Counter()
    clusterList = [] #list of length k, clusterList[0] = a counter of how many data points of each emotion type were clustered to group 0
    for i in xrange(0, k): clusterList.append(collections.Counter())
    print len(clusterAssignments)
    for i in range(0, len(clusterAssignments)):
        clusterIndex = clusterAssignments[i]
        emotion = data[i][1]
        emotionCounter[emotion] += 1
        clusterList[clusterIndex][emotion] += 1

    clusterPercentList = []##list of length k, clusterPercentList[clusterIndex] = dictionary (key: emotion --> value: percent of this emotion's total data points assigned to this cluster)
    for i in xrange(0, k): clusterPercentList.append(dict())
    for clusterIndex in range(0, k):
        counter = clusterList[clusterIndex]
        for emotion, count in counter.items():  clusterPercentList[clusterIndex][emotion] = float(count)/float(emotionCounter[emotion])


    #print "cluster list: ", clusterList
    #print "emotionCounter: ", emotionCounter
    #print "clusterPercentList: ", clusterPercentList

 
    '''
    ------------------------------------------
    APPROACH 1: 
    assign a cluster to the emotion that is most represented (based on count) in that cluster
    -----
    '''
    '''kmeansClusterIndex_to_emotion = dict()
    

    kmeansClusterIndex_to_emotion = dict()
    for i in range(0,k):
        if len(clusterList[i].most_common(1)) == 0: kmeansClusterIndex_to_emotion[i] = None
        else: kmeansClusterIndex_to_emotion[i] = (clusterList[i].most_common(1))[0][0]
        #kmeansClusterIndex_to_emotion[clusterIndex] --> emotion

    #print "kmeansClusterIndex --> emotion: ", kmeansClusterIndex_to_emotion
    return kmeansClusterIndex_to_emotion'''

    '''
    -----
    END APPROACH 1
    -----------------------------------------
    '''


    '''
    -----------------------------------------
    APPROACH 2: 
    assign an emotion to the cluster containing the highest percent of the emotion's data points
    -----
    '''
    
    #determine assignment by assigning an emotion to the cluster containing the highest percent of the emotion's data points
    
    kmeansClusterIndex_to_emotion = dict()

    #assign emotions to clusters in order of least prevelant emotion first

        #work around code to sort emotions by count
    emotionCounterRevItems = [(count, emotion) for emotion, count in emotionCounter.items()]
    sortedEmotions = sorted(emotionCounterRevItems)
    print "sortedEmotions: ", sortedEmotions


    for count, emotion in sortedEmotions:
        highestPercent = None
        bestCluster = None
        for clusterIndex in range(0, len(clusterPercentList)):
            if clusterIndex in kmeansClusterIndex_to_emotion: continue #don't want to assign two different emotions to the same cluster
            percentCounter = clusterPercentList[clusterIndex]
            if emotion not in percentCounter: continue
            if bestCluster == None or percentCounter[emotion] > highestPercent:
                bestCluster = clusterIndex
                highestPercent = percentCounter[emotion]
        kmeansClusterIndex_to_emotion[bestCluster] = emotion

    print "kmeansClusterIndex--> emotion: ", kmeansClusterIndex_to_emotion
    return kmeansClusterIndex_to_emotion

    '''
    -----
    END APPROACH 2
    --------------------------------------------
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
    
    #///////////////////// flags ///////////////////
    extractor = "surf" #"surf" # or "sift" # use this flag to set the feature extractor
    drawImage = True
    normalize = False
    kmeanstype = None #"concat" # or "first" or "independant" # use this flag to determine how to handle features for kmeans
    algorithm = "neighbours" # kmeans or "neighbours"
   #////////////////////////////////////////////////
    numCorrect = 0.0
    totalNum = 0.0

    pixelList = [pixels for pixels, emotion in testing_data1]
    
    featureToImageMap = []
    
    surfFeaturesList = []
    if kmeanstype == "concat":
        surfFeaturesList = [[]]*len(pixelList)
    #for x in range(1):
    for x in range(len(pixelList)):
        row = []
        twoDArray = []


        for i in range(0, len(pixelList[x])):
            if i % 48 == 0 and i!= 0:
                twoDArray.append(row)
                row = []
            row.append(pixelList[x][i])
        twoDArray.append(row)
        
        spoints = None
        if extractor == "sift":
            sift = cv2.SIFT()
            spoints = sift.detectAndCompute(np.uint8(np.array(twoDArray)), None)
        elif extractor == "surf":
            surf = cv2.SURF(0)   
            spoints = surf.detectAndCompute(np.uint8(np.array(twoDArray)), None)
        elif extractor == "fast":
            image = np.array(twoDArray, dtype=np.uint8)
            fast = cv2.FastFeatureDetector()
            kp = fast.detect(image)
            freak = cv2.DescriptorExtractor_create('SURF')
            spoints = freak.compute(image,kp)   


        if drawImage:
            img2 = cv2.drawKeypoints(np.uint8(np.array(twoDArray)),spoints[0],None,(255,0,0),4)
            plt.imshow(img2),plt.show()
        
        if algorithm == "neighbours":
            assignment = nearestNeighbour(twoDArray, spoints, training_data, extractor)
            totalNum += 1
            if assignment == testing_data1[x][1]:
                #print "correct"
                numCorrect +=1
           # else: 
               #print "incorrect"

        # if normalize: # WHAT DOES THIS DO???????????
        #     covar = np.cov(spoints[1], rowvar=0)
        #     covar.shape()
        #     invcovar = np.linalg.inv(covar.reshape((1,1)))
        #     invcovar = np.linalg.inv(covar)
        
        #only add 1st feature for simplicity with kmeans:
        if kmeanstype == "first":
            if spoints is not None and spoints[1] is not None:
                surfFeaturesList.append(list(spoints[1][0]))
            else:
                surfFeaturesList.append([])

        #add all features as independant points to kmeans:
        if kmeanstype == "independant":
            if spoints is not None and spoints[1] is not None:
                for point in spoints[1]:
                    surfFeaturesList.append(list(point))
                    featureToImageMap.append(x) #specifys that this feature maps to this image

        #add concatenate all features to a giant feature     
        if kmeanstype == "concat":
            if spoints is not None and spoints[1] is not None:
                for point in spoints[1]:
                    for i in point:
                        surfFeaturesList[x].append(i)
            # else: 
            #     surfFeaturesList[x] = [1]
    

    if algorithm == "neighbours":
        print "accuracy: ", numCorrect/totalNum
        return

    #normalize features list before kmeans
    if normalize:
        np.linalg.norm(surfFeaturesList)
        surfFeaturesList = scipy.cluster.vq.whiten(surfFeaturesList)
        #print "after",surfFeaturesList[0]

    k = 7
    maxIter = 10
    
    #print cv2.BFMatcher().match(surfFeaturesList[0], surfFeaturesList[1])
    #clusters = cv2.kmeans(np.array(surfFeaturesList), k, (cv2.TERM_CRITERIA_MAX_ITER, 10, .1), 1, cv2.KMEANS_RANDOM_CENTERS)
    #clusters, centroids = kmeansFeatures(surfFeaturesList, k, maxIter)
    #evaluateClusters(clusters, training_data, k)
    
    if algorithm == "kmeans":
        clusters, centroids = kmeans(surfFeaturesList, k, maxIter)

        if independant: 
            clusters = getActualClusters(clusters, featureToImageMap, pixelList)
    
        evaluateClusters(clusters, training_data, k)


def nearestNeighbour(image1, features, training_data, extractor):
    #print "starting nn-------------------------------------------"
    if features[1] is not None:
        minDistance = float("inf")
        bestEmotion= -1
        for pixels,emotion in training_data:
            #print "emotion", emotion           
            row = []
            twoDArray = []

            for i in range(0, len(pixels)):
                if i % 48 == 0 and i!= 0:
                    twoDArray.append(row)
                    row = []
                row.append(pixels[i])
            twoDArray.append(row)

            spoints = None
            if extractor == "sift":
                sift = cv2.SIFT()
                spoints = sift.detectAndCompute(np.uint8(np.array(twoDArray)), None)
            elif extractor == "surf":
                surf = cv2.SURF(4000)   
                spoints = surf.detectAndCompute(np.uint8(np.array(twoDArray)), None)
            elif extractor == "fast":
                image = np.array(twoDArray, dtype=np.uint8)
                fast = cv2.FastFeatureDetector()
                kp = fast.detect(image)
                freak = cv2.DescriptorExtractor_create('SURF')
                spoints = freak.compute(image,kp)   
      

            match = cv2.BFMatcher(cv2.NORM_L1).match(spoints[1], features[1])
            distances = [m.distance for m in match]
            # print "distances :-------------------------------"
            # for d in distances:
            #     print d
            d = sum(distances)
            if d < minDistance and len(distances) > 0:
                minDistance = d
                bestEmotion = emotion
                #print "updating best emotion to: ", emotion, "min distance to: ",d
                drawMatches(np.uint8(np.array(image1)), features[0], np.uint8(np.array(twoDArray)), spoints[0], match)
        #print "min distance: ", minDistance
        return bestEmotion 
    else:
        return random.randrange(0,7)

def getActualClusters(clusters, featureToImageMap, pixelList):
    '''
    takes in a the cluster assignments for independant features,
    determines which images those features correspond to, 
    assigns the image to the cluster to which most of its features are assigned,
    returns cluster list in the expected form of a dict declaring which cluster each image is assigned to
    '''
    actualClusters = [0] * len(pixelList)
    prevImage = -1
    imageAssignments = [] 
    for i in range(len(clusters)):
        image = featureToImageMap[i]
        if image == prevImage:
            imageAssignments.append(clusters[i])
        else:
            if prevImage != (-1):
                if imageAssignments == []:
                    actualClusters[prevImage] = random.randrange(0,7)
                else:    
#                 print imageAssignments
#                 print max(set(imageAssignments), key=imageAssignments.count)
                    actualClusters[prevImage] = max(set(imageAssignments), key=imageAssignments.count)
            imageAssignments = []
        prevImage = image
    return actualClusters



# def returnClosetCentroidFeatures(point, centroidsPrev):
#     '''
#     attempt to reqwrite the return closest centroid features to work with sift/surf features. currently does nothing/not used
#     ----
#     '''
#     runningMin = None
#     closetCentroidIndex = 0 #default to being the centroid at index 0
#     nCentroids = len(centroidsPrev)
#     for i in range(nCentroids):
#         centroid = centroidsPrev[i]
#         covar = np.cov(point, rowvar=0)
#         invcovar = np.linalg.inv(covar)
#        # distance = scipy.spatial.distance.mahalanobis(centroid, point, invcovar)
#         distance = len(cv2.BFMatcher().match(np.array(centroid), np.array(point)))
#         if (runningMin == None) or ((distance < runningMin)):
#             runningMin = distance
#             closetCentroidIndex = i

#         if((distance == runningMin)): closetCentroidIndex = random.choice([closetCentroidIndex, i])
#     return closetCentroidIndex


### Found on stack overflow http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
# credit to rayryeng
def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """


    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]


    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        if img1_idx < len(kp1) and img2_idx < len(kp2):

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    plt.imshow(out),plt.show()




def runSGD(training_data, testing_data):
    '''
    This function holds code to run stochastic gradient descent
    '''
    learnPredictor(training_data, testing_data, pixelIndexFeatureExtractor)

def runKmeans(training_data, testing_data):
    '''
    This function holds code to run kmeans clustering
    '''

    '''some functions might want data points represented as dictionaries'''
    #dataAsDictionaries = convertDataPointsToDictionaries(training_data):


    '''Code to convert data from list of tuples (pixels, emotion) to just list of pixels 
    (required for kmeans) '''

    trainingPixelList = [pixels for pixels, emotion in training_data]
    testingPixelList = [pixels for pixels, emotion in testing_data]
    
    '''kmeans clustering'''

    k = 7
    maxIter = 50
    clusters, centroids = kmeans(trainingPixelList, k, maxIter)
    evaluateClusters(clusters, training_data, k)

    '''use centroids to cluster test data'''
    clusters = clusterData(testingPixelList, centroids)
    evaluateClusters(clusters, testing_data, k)


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

    '''** Note: if smallDataSet = true, the value of testing_data2 will be None **'''

    testData = testing_data1

    #runSGD(training_data, testData)

    runKmeans(training_data, testData)
    #runSurf(training_data, testing_data1, testing_data2)

if __name__ == '__main__':
  main()
