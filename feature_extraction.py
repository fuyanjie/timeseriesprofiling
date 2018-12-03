#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Yanjie
#
# Created:     04/02/2015
# Copyright:   (c) Yanjie 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from scipy import stats
import os
import numpy as np
from collections import Counter

def longest_increasing_subsequence(X):
    n = len(X)
    X = [None] + X  # Pad sequence so that it starts at X[1]
    M = [None]*(n+1)  # Allocate arrays for M and P
    P = [None]*(n+1)
    L = 0
    for i in range(1,n+1):
        if L == 0 or X[M[1]] >= X[i]:
            # there is no j s.t. X[M[j]] < X[i]]
            j = 0
        else:
            # binary search for the largest j s.t. X[M[j]] < X[i]]
            lo = 1      # largest value known to be <= j
            hi = L+1    # smallest value known to be > j
            while lo < hi - 1:
                mid = (lo + hi)//2
                if X[M[mid]] < X[i]:
                    lo = mid
                else:
                    hi = mid
            j = lo

        P[i] = M[j]
        if j == L or X[i] < X[M[j+1]]:
            M[j+1] = i
            L = max(L,j+1)

    # Backtrack to find the optimal sequence in reverse order
    output = []
    pos = M[L]
    while L > 0:
        output.append(X[pos])
        pos = P[pos]
        L -= 1

    output.reverse()
    return output

def longest_dcreasing_subsequence( A ):
    m = [0] * len( A ) # starting with m = [1] * len( A ) is not necessary
    for x in range( len( A ) - 2, -1, -1 ):
        for y in range( len( A ) - 1, x, -1 ):
            if m[x] <= m[y] and A[x] > A[y]:
                m[x] = m[y] + 1 # or use m[x]+=1

    max_value = max( m )

    result = []
    for i in range( len( m ) ):
        if max_value == m[i]:
            result.append( A[i] )
            max_value -= 1

    return result

def main():
    #pre-defined constant variables
    MAX_PACKAGE_SIZE=1515.0
    K=4
    HOPPING_THRESHOLD=300
    M=4
    N=3
    MAX_LENGTH=20

    #prepare output txts
    open("feature_min.txt", "w").write("did,min\n")
    open("feature_max.txt", "w").write("did,max\n")
    open("feature_mean.txt", "w").write("did,mean\n")
    open("feature_variance.txt", "w").write("did,variance\n")
    open("feature_skewness.txt", "w").write("did,skewness\n")
    open("feature_kurtosis.txt", "w").write("did,kurtosis\n")
    for i in range(K):
        open("feature_%s_level_percentage.txt" % i, "w").write("did,percentage\n")
    open("feature_hopping_count.txt", "w").write("did,hopping_count\n")
    open("feature_LIS_len.txt", "w").write("did,lis_len\n")
    open("feature_LDS_len.txt", "w").write("did,lds_len\n")
    for i in range(N):
        open("feature_top_%s_freq_subseq.txt" % i, "w").write("did,freq\n")
    open("feature_VF_1st.txt", "w").write("did,VF_1st\n")
    open("feature_VF_2nd.txt", "w").write("did,VF_2nd\n")
    open("feature_VF_3rd.txt", "w").write("did,VF_3rd\n")
    open("feature_BF_1st.txt", "w").write("did,BF_1st\n")
    open("feature_BF_2nd.txt", "w").write("did,BF_2nd\n")
    open("feature_BF_3rd.txt", "w").write("did,BF_3rd\n")

    open("feature_time_delay_mean.txt", "w").write("did,time_delay_mean\n")
    open("feature_time_delay_var.txt", "w").write("did,time_delay_var\n")
    open("feature_time_delay_min.txt", "w").write("did,time_delay_min\n")
    open("feature_time_delay_max.txt", "w").write("did,time_delay_max\n")
    open("feature_time_delay_skewness.txt", "w").write("did,time_delay_skewness\n")
    open("feature_time_delay_kurtosis.txt", "w").write("did,time_delay_kurtosis\n")

    #prepare scanned folders
    root=r"C:\Users\fuyanjie\Dropbox\Codebase\KDD15-WeChat-exp\data"
    usages=["1_GroupPicture", "2_GroupText", "3_GroupVoice", "4_IndividualPicture", "5_IndividualStreamVideo", "6_IndividualText", "7_IndividualVoice", "8_Moment", "9_IndividualLocation", "10_IndividualSight"]
    #usages=[ "4_IndividualPicture", "5_IndividualStreamVideo", "6_IndividualText", "7_IndividualVoice", "8_Moment", "9_IndividualLocation", "10_IndividualSight"]
    folders=[r"%s\%s" % (root, usage) for usage in usages]

    #start to extract features
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                did=file[:-4]
                path=r"%s\%s" % (folder, file)
                step=list()
                time_stamp=list()
                packet_size=list()
                for idx, line in enumerate(open(path)):
                    if idx>0:
                        tokens=line.split(",")
                        if len(tokens)>=7:
                            tokens=[token.strip("\"")  for token in tokens]
                            step.append(int(tokens[0]))
                            time_stamp.append(float(tokens[1]))
                            packet_size.append(float(tokens[5]))

                if len(step)>0:
                    #packet size realted features

                    #descriptive statistics
                    nobs, (minval, maxval),  mean, variance, skewness, kurtosis=stats.describe(packet_size)
                    open("feature_min.txt", "a").write("%s,%s\n" % (did, minval))
                    open("feature_max.txt", "a").write("%s,%s\n" % (did, maxval))
                    open("feature_mean.txt", "a").write("%s,%s\n" % (did, mean))
                    open("feature_variance.txt", "a").write("%s,%s\n" % (did, variance))
                    open("feature_skewness.txt", "a").write("%s,%s\n" % (did, skewness))
                    open("feature_kurtosis.txt", "a").write("%s,%s\n" % (did, kurtosis))
                    #percentage of packages with size greater than {MAX_PACKAGE_SIZE*k/K}_{k=1}^K
                    for i in range(K):
                        k_level_percentage=sum([1 for s in packet_size if s >= (i*MAX_PACKAGE_SIZE/K) and s < ((i+1)*MAX_PACKAGE_SIZE/K)] )*1.0/nobs
                        open("feature_%s_level_percentage.txt" % i, "a").write("%s,%s\n" % (did, k_level_percentage))
                    #hopping counts of a squence
                    hopping_count=sum([1 for i in range(nobs-1) if np.abs(packet_size[i]-packet_size[i+1]) > HOPPING_THRESHOLD])
                    open("feature_hopping_count.txt", "a").write("%s,%s\n" % (did, hopping_count))
                    #length of longest increasing subsequence
                    lis_len=len(longest_increasing_subsequence(packet_size))
                    open("feature_LIS_len.txt", "a").write("%s,%s\n" % (did, lis_len))
                    #length of longest decreasing subsequence
                    lds_len=len(longest_dcreasing_subsequence(packet_size))
                    open("feature_LDS_len.txt", "a").write("%s,%s\n" % (did, lds_len))
                    #Top-N Frequent Continuous Subsequence
                    packet_string=[str(int(s*M/MAX_PACKAGE_SIZE)) for s in packet_size]
                    packet_string=''.join(packet_string)
                    substrs=list()
                    for L in range(3, (MAX_LENGTH+1)):
                        substrs=substrs+[packet_string[i:i+L] for i in range(0, len(packet_string) - (L - 1))]
                    substr_counter=Counter(substrs)
                    freq=substr_counter.values()
                    freq=sorted(freq, reverse=True)
                    for i in range(N):
                        open("feature_top_%s_freq_subseq.txt" % i, "a").write("%s,%s\n" % (did, freq[i]))
                    #Forward variance
                    var_forward_1st=np.var(packet_size[nobs/4:-1])#25%poisition
                    var_forward_2rd=np.var(packet_size[nobs/2:-1])#50%poisition
                    var_forward_3nd=np.var(packet_size[nobs*3/4:-1])#75%poisition
                    open("feature_VF_1st.txt", "a").write("%s,%s\n" % (did, var_forward_1st))
                    open("feature_VF_2nd.txt", "a").write("%s,%s\n" % (did, var_forward_2rd))
                    open("feature_VF_3rd.txt", "a").write("%s,%s\n" % (did, var_forward_3nd))
                    #Backward variance
                    var_backward_1st=np.var(packet_size[nobs/4:-1])#25%poisition
                    var_backward_2rd=np.var(packet_size[nobs/2:-1])#50%poisition
                    var_backward_3nd=np.var(packet_size[nobs*3/4:-1])#75%poisition
                    open("feature_BF_1st.txt", "a").write("%s,%s\n" % (did, var_backward_1st))
                    open("feature_BF_2nd.txt", "a").write("%s,%s\n" % (did, var_backward_2rd))
                    open("feature_BF_3rd.txt", "a").write("%s,%s\n" % (did, var_backward_3nd))

                    #Delay Time Related Features
                    time_delay=[time_stamp[i+1]-time_stamp[i] for i in range((nobs-1))]
                    time_delay_nobs, (time_delay_minval, time_delay_maxval),  time_delay_mean, time_delay_variance, time_delay_skewness, time_delay_kurtosis=stats.describe(time_delay)

                    open("feature_time_delay_mean.txt", "a").write("%s,%s\n" % (did, time_delay_mean))
                    open("feature_time_delay_var.txt", "a").write("%s,%s\n" % (did, time_delay_variance))
                    open("feature_time_delay_min.txt", "a").write("%s,%s\n" % (did, time_delay_minval))
                    open("feature_time_delay_max.txt", "a").write("%s,%s\n" % (did, time_delay_maxval))
                    open("feature_time_delay_skewness.txt", "a").write("%s,%s\n" % (did, time_delay_skewness))
                    open("feature_time_delay_kurtosis.txt", "a").write("%s,%s\n" % (did, time_delay_kurtosis))


if __name__ == '__main__':
    main()
