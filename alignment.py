import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    a_index = 0 # for keeping track of our place in the a matrix (since we add rows two at a time)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN

        row1 = [a_x, a_y, 1, 0, 0, 0, -b_x*a_x, -b_x*a_y, -b_x] # fill the first row for the match
        row2 = [0, 0, 0, a_x, a_y, 1, -b_y*a_x, -b_y*a_y, -b_y] # fill the second row for the match

        # place the rows in the table
        A[a_index] = row1
        A[a_index+1] = row2

        a_index += 2

        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN

    #H = (Vt[Vt.shape[0] - 1]/Vt[Vt.shape[0] - 1][8]).reshape(3,3) # set homography to smallest right singular vector (the last row vector, normalized)
    H = Vt[-1].reshape(3,3)

    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.

    #TODO-BLOCK-BEGIN
    # figure out the minimal number of matches needed to align
    minMatches = 0

    if(m == eTranslate):
        minMatches = 1
    elif (m == eHomography):
        minMatches = 4

    maxInliers = 0
    best_inliers = []

    for iteration in range(nRANSAC):
        # sample the necessary number of matches
        matchSample = random.sample(matches, minMatches)

        H_estimate = np.eye(3,3)

        # estimate the homography
        if(m == eHomography):
            H_estimate = computeHomography(f1, f2, matchSample)
        else:
            # calculate translations
            translationX = f2[matchSample[0].trainIdx].pt[0] - f1[matchSample[0].queryIdx].pt[0]
            translationY = f2[matchSample[0].trainIdx].pt[1] - f1[matchSample[0].queryIdx].pt[1]

            # set the translate fields in the H_estimate
            H_estimate[0, 2] = translationX
            H_estimate[1, 2] = translationY

        # calculate the inliers for the ting
        inliers = getInliers(f1, f2, matches, H_estimate, RANSACthresh)

        # if the number if inliers is higher than previous iterations, update the best estimates
        if len(inliers) > maxInliers:
            maxInliers = len(inliers)
            best_inliers = inliers

    # calculate the least squares fit for the best motion estimate
    M = leastSquaresFit(f1, f2, matches, m, best_inliers)

    #TODO-BLOCK-END
    #END TODO

    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN

        queryInd = matches[i].queryIdx
        trainInd = matches[i].trainIdx

        queryPoint = np.array([f1[queryInd].pt[0],  f1[queryInd].pt[1], 1]).T # make the query point
        transformedQueryFeature = M.dot(queryPoint) # transform the query feature by the transformation matrix

        # assemble new point/feature for comparison
        comp1 = [transformedQueryFeature[0]/transformedQueryFeature[2], transformedQueryFeature[1]/transformedQueryFeature[2]]
        comp2 = np.array(f2[trainInd].pt)[:2]

        if(np.linalg.norm(comp1-comp2) <= RANSACthresh):
            inlier_indices.append(i)

        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN

            point1 = f1[matches[inlier_indices[i]].queryIdx].pt
            point2 = f2[matches[inlier_indices[i]].trainIdx].pt

            u += point2[0] - point1[0] # compute x distance and add to the vector sum
            v += point2[1] - point1[1] # compute y distance and add to the vector sum

            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        inlier_matchset = []
        for i in range(len(inlier_indices)):
            inlier_matchset.append(matches[inlier_indices[i]])

        M = computeHomography(f1, f2, inlier_matchset)
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M
