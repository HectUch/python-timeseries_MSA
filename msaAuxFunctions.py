#--------------------------------------------------------------------------------------------
#Description: File with the implementations of Functions that aid the use of trs and MSA, including imports
#Version    : v2.0
#Author     : Heitor Uchoa
#Note : All the auxiliary functions will be here
#--------------------------------------------------------------------------------------------

from audioop import avg, mul
from http.client import FORBIDDEN
from pickle import TRUE
import sys
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import PiecewiseAggregateApproximation
from trsfile.parametermap import TraceSetParameterMap
import trsfile.traceparameter as tp
import trsfile
import sys
import re
import os
import random, os
from trsfile import trs_open, Trace, SampleCoding, TracePadding, Header
from trsfile.parametermap import TraceParameterMap, TraceParameterDefinitionMap
from trsfile.traceparameter import ByteArrayParameter, ParameterType, TraceParameterDefinition
import seaborn as sns; sns.set_theme()
import matplotlib.gridspec as gs
import time
import copy
import pandas as pd


nucleotide_alphabet = ['c','s','t','a','g','p','d','e','q','n','h','r','k','m','i','l','v','w','y','f']
#nucleotide_alphabet = ['Y', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'V', 'W', 'A']
dna_alphabet = ['A','M','C','S','G','K','T']
in_path = 'Data\source_traces'
out_path = 'Data\\result_traces'
fasta_in = 'Data\\fasta_in'
fasta_out = 'Data\\fasta_out'
end_txt = '.txt'
options = ' --globalpair '#'--localpair '#'--globalpair ' #'--localpair ' #

def step_minmaxTraces(trace_set,len_alphabet) :
    min_v = 127
    max_v = -128
    step = 12.75
    for trace in trace_set:
        a = min(trace)
        b = max(trace)
        if min_v > a :
            min_v = a
        if max_v < b :
            max_v = b
    step = (int(abs(max_v)) + int(abs(min_v)))/len_alphabet
    print(max_v)
    print(min_v)
    return step

def convert2MinMaxnucleotides(trace_set,len_alphabet=19,n_samples=1000):
  
    ms_unaligned = []
    nucleotides_seq = []
    step = 12.75
    for index,item in enumerate(trace_set):
        step = (max(trace_set[index])-min(trace_set[index]))/(len_alphabet)
        for j,k in enumerate(trace_set[index]):
            if j == n_samples :
                break
            nIndex = round(trace_set[index][j]/step)
            if nIndex > len_alphabet:
                nIndex = len_alphabet
            if nIndex < 0:
                nIndex = 0
            nucleotides_seq.append(nucleotide_alphabet[nIndex])            
        ms_unaligned.append(nucleotides_seq)        
        nucleotides_seq = []        
    
    return ms_unaligned

def compressWSymbolsMax(msa_un,maxCompression = -1):
    #Add an extra vector with the number of values per compressed caracter or the interval they represent , ex: A : 1 - 50
    grouped = []
    lin = []
    quant = -1
    wordsPerSymbol = []
    compressionMatrix = []
    lastWord = ''
    for i in msa_un:
        for j in i:
            quant += 1
            if (j != lastWord or ((quant == maxCompression) and (maxCompression != -1))):
                if (lastWord != '') :
                    wordsPerSymbol.append(quant)
                    quant = 0
                grouped.append(j)
                lastWord = j                                
        lin.append(grouped)
        wordsPerSymbol.append(quant+1)
        compressionMatrix.append(wordsPerSymbol)
        wordsPerSymbol = []
        grouped = []
        quant = 0 
        lastWord = ''        
    return [lin,compressionMatrix]

def compressWSymbolsGrad(msa_un,maxCompression = -1):
    #Add an extra vector with the number of values per compressed caracter or the interval they represent , ex: A : 1 - 50
    grouped = []
    lin = []
    quant = -1
    wordsPerSymbol = []
    compressionMatrix = []
    lastWord = ''    
    for i in msa_un:
        for j in i:
            quant += 1
            if (j != lastWord or ((quant == maxCompression) and (maxCompression != -1))):
                if (lastWord != '') :
                    wordsPerSymbol.append(quant)
                    quant = 0
                    if nucleotide_alphabet.index(lastWord) > nucleotide_alphabet.index(j):
                        grouped.append('W')
                        wordsPerSymbol.append(1)
                    else:
                         grouped.append('A')
                         wordsPerSymbol.append(1)
                grouped.append(j)
                lastWord = j                                
        lin.append(grouped)
        wordsPerSymbol.append(quant+1)
        compressionMatrix.append(wordsPerSymbol)
        wordsPerSymbol = []
        grouped = []
        quant = 0 
        lastWord = ''        
    return [lin,compressionMatrix]

def decompress_gapsGradV2(msa_zip,compressionRef):
    #create a function here with Max lenght MSA_s
    #print(compressionRef) 
    maxPerIndex = findMaxLenghtPerIndexV2(msa_zip,compressionRef)
    decompress_traces = []
    traces_Aligned = []
    #n_per_trace = [0] * len(compressionRef[0]) #n_samples should be equal to this size
    n = 0 
      
    for i in range(0,len(msa_zip)):        
        for j in  range(0,len(msa_zip[0])):
            #print('MSA: ' + msa_zip[i][j] + ' ' +  str(j) + ' ' + str(i))
            if msa_zip[i][j] != '-':
                refQuantity = compressionRef[i][n]
                decompress_traces = decompress_traces + ([msa_zip[i][j]] * refQuantity) + (['-']*(maxPerIndex[j]-refQuantity))
                n += 1 
            else:
                decompress_traces = (['-']*(maxPerIndex[j]))+ decompress_traces #+  (['-']*(maxPerIndex[j]))
        n = 0
        traces_Aligned.append(decompress_traces)
        decompress_traces = []
    
    
    return traces_Aligned

def convert2nucleotides(trace_set,len_alphabet=19,n_samples=1000,step=12.75):
  
    ms_unaligned = []
    nucleotides_seq = []
    for index,item in enumerate(trace_set):
        for j,k in enumerate(trace_set[index]):
            if j == n_samples :
                break
            nIndex = round((trace_set[index][j]+127)/step)
            if nIndex > len_alphabet:
                nIndex = len_alphabet
            if nIndex < 0:
                nIndex = 0
            nucleotides_seq.append(nucleotide_alphabet[nIndex])
        ms_unaligned.append(nucleotides_seq)        
        nucleotides_seq = []
    
    return ms_unaligned

def convert2dna(trace_set,len_alphabet=6,n_samples=1000,step=42.34, dna_alphabet = ['a','c','t','g']):
    
    dna_unaligned = []
    dna_seq = []
    for index,item in enumerate(trace_set):
        for j,k in enumerate(trace_set[index]):
            if j == n_samples :
                break
            nIndex = round((trace_set[index][j]+127)/step)
            if nIndex > len_alphabet:
                nIndex = len_alphabet
            if nIndex < 0:
                nIndex = 0
            dna_seq.append(dna_alphabet[nIndex])            
        dna_unaligned.append(dna_seq)        
        dna_seq = []
        
    
    return dna_unaligned

def selectWindow(trace_set,start,samples,n_traces):
    window = []
    goal_trace_set = []
    for i in range(0,n_traces):
        for j in range(start,start+samples):           
            window.append(trace_set[i][j])
        goal_trace_set.append(window)
        window = []    
    return goal_trace_set

def selectWindow2(trace_set,start,samples,n_t_start,n_traces):
    window = []
    goal_trace_set = []
    for i in range(n_t_start,n_traces):
        for j in range(start,start+samples):           
            window.append(trace_set[i][j])
        goal_trace_set.append(window)
        window = []    
    return goal_trace_set

def copy_e(trace_set):
    window = []
    goal_trace_set = []
    for i in range(0,len(trace_set)):
        window = window + trace_set[i][:]
        goal_trace_set.append(window)
        window = []    
    return goal_trace_set

def mergeTraceAt(trace_set,trace2fit,start,samples,n_traces):
    window = []
    goal_trace_set = []
    j = 0
    for i in range(0,n_traces):
        while j < 30000:
            if (j < start) or (j > start+samples) :           
                window.append(trace_set[i][j])
            else:
                for k in range(len(trace2fit[0])):
                    window.append(trace2fit[i][k])
                j = (start+samples-1)
        goal_trace_set.append(window)
        window = []    
    return goal_trace_set

def mergeTraceAt(trace_set,trace2fit,start,samples,n_traces):
    
    goal_trace_set = []
    j = 0
    for i in range(0,n_traces):          
                goal_trace_set.append([*trace_set[i][0:start],*trace2fit[i],*trace_set[i][start+samples:]])

    return goal_trace_set

def print_msa(ms_unAligned,file_msa,n_samples) :
    msa = ''
    cnt = 0   
    for index,seq in enumerate(ms_unAligned) :
        msa += '> Trace' + str(index) + '\n'
        for cnt,j in enumerate(ms_unAligned[index][:n_samples]) :
            msa += j.upper()
            if ((cnt + 1) % 16 == 0) :
                msa += '\n'
        msa += '\n'
    msa += '\n'
    
    with open(file_msa, 'w') as f:
        print(msa, file=f)
    
    return msa

def print_msa_singleLine(ms_unAligned,file_msa,n_samples) :
    msa = ''
    cnt = 0   
    for index,seq in enumerate(ms_unAligned) :
        msa += '> Trace' + str(index) + '\n'
        for cnt,j in enumerate(ms_unAligned[index][:n_samples]) :
            msa += j.upper()
            #if ((cnt + 1) % 16 == 0) :
            #    msa += '\n'
        msa += '\n'
    msa += '\n'
    
    with open(file_msa, 'w') as f:
        print(msa, file=f)
    
    return msa

def readMSA(file):
    
    alignedMSA = []    
    msaA = ''
    f = open(file,'r') 
    for x in f:
        if x.find('>') != -1 :
            alignedMSA.append(msaA)
            msaA = ''            
        else:
            msaA += x.replace('\n','')    
    alignedMSA.append(msaA)  
    alignedMSA.remove('')  
    f.close()    
    return alignedMSA


def readText(file):
    
    alignedMSA = []    
    msaA = ''
    f = open(file,'r') 
    for x in f:
        if x.find('>') != -1 :
            alignedMSA.append(msaA)
            msaA = ''            
        else:
            msaA += x.replace('\n','')    
    alignedMSA.append(msaA)  
    alignedMSA.remove('')  
    f.close()    
    return alignedMSA

def paamsa2trace(refTraces,msaAligned,window):
     #Will different sizes arrays encounter problems?
    traceFromMsa = []
    traceAligned = []
    n = 0   #points to previous value of trace
    gap = 0 #np.nan    
    for index, seqAA in enumerate(msaAligned) :
        n = 0            
        for aa in seqAA :
            if aa == '-':
                traceAligned.append(gap)                
            else:
                for i in range((n*window), ((n*window)+window-1)):
                    traceAligned.append(refTraces[index][i])
                n += 1               
        traceFromMsa.append(traceAligned)
        traceAligned = []            
    # It can mess up the convertion back!                  
    return traceFromMsa
   
def print_corr_heat(trace_set,file_a):
    
    traces_result = trsfile.open(file_a, 'r')    
    staticAlign = trsfile.open('noisy2000tracesRegion + StaticAlign.trs', 'r')  #    
    corr_orig = np.corrcoef(trace_set)
    corr_final = np.corrcoef(traces_result)
    corr_static = np.corrcoef(staticAlign)
    fig = plt.figure()
    N_rows_a, _ = corr_orig.shape
    N_rows_b, _ = corr_final.shape
    grid=gs.GridSpec(2,2, height_ratios=[N_rows_a,N_rows_b], width_ratios=[50,1])
    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[1,0], sharex=ax1)
    ax3 = fig.add_subplot(grid[2,0], sharex=ax1)
    cax = fig.add_subplot(grid[:,1])

    ax1.set_title('Pairwise correlation : Original Traces')
    #ax2.set_title('Pairwise correlation : MAFFT MSA Aligned Traces ')
    ax2.set_title('Pairwise correlation : Static Aligned Traces')

    mask = np.triu(corr_orig)
    mask1 = np.triu(corr_final)
    mask2 = np.triu(corr_static)          

    #fig, (ax1, ax2) = plt.subplots(ncols=2)
    #anx = sns.heatmap(corr_orig)
    #anx1 = sns.heatmap(corr_final)
    sns.heatmap(corr_orig,vmin=0.6,annot=True, cmap="inferno",mask=mask, ax=ax1, cbar_ax=cax,linewidths=.5)
    #sns.heatmap(corr_final,vmin=0.6,annot=True, cmap="inferno",mask=mask1, ax=ax2, cbar_ax=cax,linewidths=.5)
    sns.heatmap(corr_static,vmin=0.6,annot=True, cmap="inferno",mask=mask2, ax=ax2, cbar_ax=cax,linewidths=.5)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.show()
    return 0

def printToText(ms_unAligned,file_msa,n_samples):
    msa = ''
    cnt = 0   
    for index,seq in enumerate(ms_unAligned) :
        msa += '> Trace' + str(index) + '\n'
        for cnt,j in enumerate(ms_unAligned[index][:n_samples]) :
            msa += j.upper()
            if ((cnt + 1) % 16 == 0) :
                msa += '\n'
        msa += '\n'
    msa += '\n'
    
    with open(file_msa, 'w') as f:
        print(msa, file=f)
    
    return msa

def closeGapWAvarege(traces,msa):
    
    col = len(traces)
    samples = len(traces[0])
    avg = 0
    sum = 0
    gaps= 0
    
    for i in range(0,samples):
        gaps = 0
        avg = 0
        sum = 0
        for j in range(0,col):
            if msa[j,i] == '-':
                gaps += 1
            else:
                sum += traces[j,i]
        if gaps > 0:
            avg = (sum/(col-gaps))
            for j in range(0,col):
                if msa[j,i] == '-':
                    traces[j,i] = avg
    
    return traces         

def closeGap(gapped,step,begin,end):  
    
    for i in range(begin,end):
        gapped[i] = gapped[i-1] + step
    
def averageMethod(gappedTrace):
    
    connectedTrace = gappedTrace
    size = len(connectedTrace)
    invalid = 0 # np.nan
    #cases : 
    # 1-trace starts with zero
    n = 0     
    if connectedTrace[0] == invalid:
        for i in range(size):
            if connectedTrace[i] != invalid:
                n = i                
                break
        for i in range(n):    
            connectedTrace[i] = connectedTrace[n]
    # 2-zero in between numbers / Search gaps
    begin = -1
    for i in range(size):
        if i == 0:
            continue
        if connectedTrace[i] == invalid and connectedTrace[i-1] != invalid :
               begin = i               
        if connectedTrace[i] != invalid and connectedTrace[i-1] == invalid :
               step = abs(connectedTrace[i] - connectedTrace[begin-1])/((i - begin)+1)
               closeGap(connectedTrace,step,begin,i)
               begin = -1
                   
    # 3-trace finishes in zero
    if begin != -1:
        for i in range(begin,size):
            connectedTrace[i] = connectedTrace[begin-1]
    
    return connectedTrace

def msa2trace(refTraces,msaAligned):
    
    traceFromMsa = []
    traceAligned = []
    n = 0 # points to previous value of trace
    gap = 0#np.nan    
    for index, seqAA in enumerate(msaAligned) :
        n = 0            
        for aa in seqAA :            
            if aa == '-':
                traceAligned.append(gap)                
            else:
                traceAligned.append(refTraces[index][n])
                n += 1             
                              
        traceFromMsa.append(traceAligned)
        traceAligned = []            
    # It can mess up the convertion back!                  
    return traceFromMsa

def msa2tracev2(refTraces,msaAligned):
    
    traceFromMsa = []
    traceAligned = []
    n = 0 # points to previous value of trace
    gap = 0#np.nan    
    for index, seqAA in enumerate(msaAligned) :
        n = 0            
        for aa in seqAA :
            #print('My guy :' + aa)
            if aa == '-':                
                traceAligned.append(gap)                
            else:
                traceAligned.append(refTraces[index][n])
                n += 1                
                if n >= len(refTraces[0]):
                    break
                #print('My last N : ' + str(n))               
        traceFromMsa.append(traceAligned)
        traceAligned = []            
    #It can mess up the convertion back!                  
    return traceFromMsa

def trace2text(trace_set,n_samples=1000):
  
    tx_unaligned = []
    tx_subseq = []
    not_allowed = [0x3C,0x3D,0x3E,0x2D,0x20,0x0D,0x0A,0x00]
    
    for index in range(0,len(trace_set)):
        for j in range(0,n_samples):
            if j == n_samples:
                break
            sample = abs((trace_set[index][j]))
            if sample in not_allowed:
                if sample == 60 :
                    sample = 59 
                elif ((sample == 61)):
                    sample = 63
                else:
                    sample += 1                
            tx_subseq.append(chr(sample))
        tx_unaligned.append(tx_subseq)
        tx_subseq = []
        
    return tx_unaligned
    
def text2trace(msaAligned,refTraces):
    
    traceFromMsa = []
    traceAligned = []
    gap = -120 #np.nan
    n = 0
    mult = 1         
    for index, seqAA in enumerate(msaAligned):
        n = 0                   
        for aa in seqAA:
            if aa == '-':
                traceAligned.append(gap)                
            else:
                if refTraces[index][n] >= 0:
                    mult = 1
                else:
                    mult = -1
                traceAligned.append(ord(aa)*mult)
                n +=1                               
        traceFromMsa.append(traceAligned)
        traceAligned = []   
                     
    return traceFromMsa

def print_trs_diff(trace_src,to_print,file_name,total_traces):       
    headers =  trace_src.get_headers()   
    with trs_open(
        file_name+'diff.trs',                 # File name of the trace set
        'w',                             # Mode: r, w, x, a (default to x)
        # Zero or more options can be passed (supported options depend on the storage engine)
        #engine = 'TrsEngine',            # Optional: how the trace set is stored (defaults to TrsEngine)
        headers = {
                   Header.TRS_VERSION : 2,
                   Header.SCALE_X: headers[Header.SCALE_X] ,
                   Header.SCALE_Y: headers[Header.SCALE_Y] ,
                   Header.LABEL_X : headers[Header.LABEL_X],
                   Header.LABEL_Y : headers[Header.LABEL_Y],
                   Header.LENGTH_DATA : headers[Header.LENGTH_DATA]                         
                }
    ) as tracese:
        for i in range(0, total_traces):
            tracese.append(
                Trace(
                    SampleCoding.BYTE,
                    np.diff(to_print[i]),
                    TraceParameterMap({'parameter' : trace_src[i].parameters['LEGACY_DATA']})
                )
            )
    return

def print_trs(trace_src,to_print,file_name,total_traces):       
    headers =  trace_src.get_headers()   
    with trs_open(
        file_name+'.trs',                 # File name of the trace set
        'w',                             # Mode: r, w, x, a (default to x)
        # Zero or more options can be passed (supported options depend on the storage engine)
        #engine = 'TrsEngine',            # Optional: how the trace set is stored (defaults to TrsEngine)
        headers = {
                   Header.TRS_VERSION : 2,
                   Header.SCALE_X: headers[Header.SCALE_X] ,
                   Header.SCALE_Y: headers[Header.SCALE_Y] ,
                   Header.LABEL_X : headers[Header.LABEL_X],
                   Header.LABEL_Y : headers[Header.LABEL_Y],
                   Header.LENGTH_DATA : headers[Header.LENGTH_DATA]                         
                }
    ) as tracese:
        for i in range(0, total_traces):
            tracese.append(
                Trace(
                    SampleCoding.BYTE,
                    to_print[i],
                    TraceParameterMap({'parameter' : trace_src[i].parameters['LEGACY_DATA']})
                )
            )
    return

def print_trs_counterFind(trace_src,consensus,to_print,file_name,total_traces,window,threshold):
    headers =  trace_src.get_headers()
    isCounter = counterMeasurePrediction(consensus,window,threshold)   
    with trs_open(
        file_name+'.trs',                 # File name of the trace set
        'w',                             # Mode: r, w, x, a (default to x)
        # Zero or more options can be passed (supported options depend on the storage engine)
        #engine = 'TrsEngine',            # Optional: how the trace set is stored (defaults to TrsEngine)
        headers = {
                   Header.TRS_VERSION : 2,
                   Header.SCALE_X: headers[Header.SCALE_X] ,
                   Header.SCALE_Y: headers[Header.SCALE_Y] ,
                   Header.LABEL_X : headers[Header.LABEL_X],
                   Header.LABEL_Y : headers[Header.LABEL_Y],
                   Header.LENGTH_DATA : headers[Header.LENGTH_DATA]                         
                }
    ) as tracese:
        tracese.append(
                Trace(
                    SampleCoding.BYTE,
                    isCounter
                )
            )
        for i in range(0, total_traces):
            tracese.append(
                Trace(
                    SampleCoding.BYTE,
                    to_print[i],
                    TraceParameterMap({'parameter' : trace_src[i].parameters['LEGACY_DATA']})
                )
            )    
    return

def print_trs_consensus(trace_src,consensus,to_print,file_name,total_traces):       
    headers =  trace_src.get_headers()   
    with trs_open(
        file_name+'.trs',                 # File name of the trace set
        'w',                             # Mode: r, w, x, a (default to x)
        # Zero or more options can be passed (supported options depend on the storage engine)
        #engine = 'TrsEngine',            # Optional: how the trace set is stored (defaults to TrsEngine)
        headers = {
                   Header.TRS_VERSION : 2,
                   Header.SCALE_X: headers[Header.SCALE_X] ,
                   Header.SCALE_Y: headers[Header.SCALE_Y] ,
                   Header.LABEL_X : headers[Header.LABEL_X],
                   Header.LABEL_Y : headers[Header.LABEL_Y],
                   Header.LENGTH_DATA : headers[Header.LENGTH_DATA]                         
                }
    ) as tracese:
        tracese.append(
                Trace(
                    SampleCoding.BYTE,
                    consensus
                )
            )
        for i in range(0, total_traces):
            tracese.append(
                Trace(
                    SampleCoding.BYTE,
                    to_print[i],
                    TraceParameterMap({'parameter' : trace_src[i].parameters['LEGACY_DATA']})
                )
            )
    return

def covTxt2Trs(traces,output,file_inspector,n_traces,initial,n_samples,alphabet_size = 20,method = 1,connect = 2, mergeBack = 0):
    
    #Parameters
    #n_traces = 500
    #initial = 361203
    #n_samples = 20000 
    #alphabet_size = 20
    #---------------------------------------------
    method = 1 # 1:y-axis, 2:SaX and 3:PAA 4:textAlign
    connect = 2 # 1:with gaps, 2:avg gaps, 3:column avg gap 
    mergeBack = 0 # 1:yes, others:No
    #---------------------------------------------
    #Traces saved : #'goodRegionTraceExtract.trs' #'noisy2000tracesRegion.trs'#'TEST_set_good-misaligned1000samples.trs' 'Intentional_-misaligned10ksamples.trs' #'difficultRegion5k.trs'
    #Internals
    in_path = 'Data\source_traces'
    out_path = 'Data\\result_traces'
    fasta_in = 'Data\\fasta_in'
    fasta_out = 'Data\\fasta_out'
    end_txt = '.txt'

    file_trace = in_path + '\good-misaligned1000samples.trs'
    #file_msa =  '\msa_preAlign' + str(n_traces) + 't_' + str(n_samples) + 's_'
    #file_result = '\msa_postAlign' + str(n_traces) + 't_' + str(n_samples) + 's_'
    file_inspector = out_path + '\MAFFT_' + str(n_traces) + 't_' + str(n_samples) + 's_' 
    nucleotide_alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    input = fasta_in + '\msa_preAlign' + str(n_traces) + 't_' + str(n_samples) + 's_' + end_txt
    output = fasta_out + '\msa_postAlign400t_3000s_2'#\msa_postAlign' + str(n_traces) + 't_' + str(n_samples) + 's_' + end_txt
    base_trace = file_inspector + '_src.trs'
    mafft = 'mafft --anysymbol --localpair --maxiterate 1000 ' + input  + ' > ' + output

    traces = trsfile.open(file_trace, 'r')
    desired_traces = selectWindow(traces,initial,n_samples,n_traces)

    start = time.time()

    step = step_minmaxTraces(desired_traces,alphabet_size)

    #Y-axis
    if method == 1:
        ms_unaligned = convert2nucleotides(desired_traces,alphabet_size-1,n_samples,step)
        file_inspector += '_yD'
    #SaX
    elif method == 2:
        sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal',alphabet=nucleotide_alphabet[:alphabet_size])
        ms_unaligned = sax.fit_transform(desired_traces)    
        file_inspector += '_SaX'
    # PAA transformation
    elif method == 3:
        window_size = 5
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        paa_trace = paa.transform(desired_traces)
        ms_unaligned = convert2nucleotides(paa_trace,alphabet_size,(n_samples/window_size),step-1)
        file_inspector += '_PAA'
    #Text
    else :
        file_inspector += '_txt'
        ms_unaligned = trace2text(desired_traces,n_samples)
        mafft = 'mafft --text ' + input + ' > ' + output
    
    traces_with_gaps = readMSA(output)

    if method == 3:
        traces_bioaligned = paamsa2trace(desired_traces,traces_with_gaps,window_size)
    elif method == 4:
        traces_bioaligned =  text2trace(traces_with_gaps,desired_traces)   
    else:
        traces_bioaligned = msa2trace(desired_traces,traces_with_gaps)

    end = time.time()

    total_time = end-start
    print("My running time in seconds was:\n"+str(total_time))

    

    if connect == 1 :
        file_inspector += '_gap'
    elif connect == 2:
        for i in range(0, n_traces):
            traces_bioaligned[i] = averageMethod(traces_bioaligned[i])
        file_inspector += '_avg'
    elif connect == 3 :
        traces_bioaligned = closeGapWAvarege(traces_bioaligned,traces_with_gaps)
        file_inspector += '_avgCol'

    if mergeBack == 1:
        traces_bioaligned = mergeTraceAt(traces,traces_bioaligned,initial,n_samples,n_traces)
        file_inspector += '_onSource.trs'
    else:
        file_inspector += '.trs'

    #Print final and Original section
    headers = traces.get_headers()
    print_trs(headers,traces_bioaligned,file_inspector,n_traces)
    print_trs(headers,desired_traces,base_trace,n_traces)
        
        
    return

def countLeftGaps(string_with_gaps,pos):
    shiftLeft = 0
    n = pos
    while n > 0:
        if string_with_gaps[n] == '-':
            shiftLeft += 1
        n -= 1
    return shiftLeft

def findLongest(string_without_gaps):
    
    begin = 0
    end = 0
    
    beginMax = 0
    endMax = 0
    for i in range(1,len(string_without_gaps)):
       if string_without_gaps[i] == '-' and  string_without_gaps[i-1] != '-':
           end = i -1
           if (end - begin) > (endMax - beginMax):
               endMax = end
               beginMax = begin
       if string_without_gaps[i] != '-' and  string_without_gaps[i-1] == '-':
           begin = i         
           
    
    return [beginMax,endMax]
 
def eliminateGaps(trace_set,msa,chop_off = 0):
    no_gaps = []
    n = 0
    pos = findLongest(msa[0])
    maxShift = 0
    for i in msa:
        #pos = findLongest(i)
        shift = countLeftGaps(i,pos[1])
        if shift > maxShift :
            maxShift = shift        
        no_gaps.append([0]*shift + trace_set[n])#(trace_set[n][shift-1:]+trace_set[n][0:shift])# Doing this wont work on small traces
        n += 1
    if chop_off == 1:
        cut = []
        for i in no_gaps:
            cut.append(i[maxShift:])
            
        return cut
    return no_gaps

#returns no-zeros
def findConsensus(msa):
    consensus = []
    nonGaps = 0
    for j in range(0,len(msa[0])):
        for i in range(0,len(msa)):
            if msa[i][j] != '-':
                nonGaps += 1
        consensus.append(nonGaps)
        nonGaps = 0   
    return consensus    

def compressSequence(msa_un):
    #Add an extra vector with the number of values per compressed caracter or the interval they represent , ex: A : 1 - 50
    grouped = []
    lin = []    
    lastWord = ''
    for i in msa_un:
        for j in i:            
            if j != lastWord:                
                grouped.append(j)
                lastWord = j                              
        lin.append(grouped)       
        lastWord = ''        
    return lin

def compressWSymbols(msa_un):
    #Add an extra vector with the number of values per compressed caracter or the interval they represent , ex: A : 1 - 50
    grouped = []
    lin = []
    quant = -1
    wordsPerSymbol = []
    compressionMatrix = []
    lastWord = ''
    for i in msa_un:
        for j in i:
            quant += 1
            if j != lastWord:
                if lastWord != '' :
                    wordsPerSymbol.append(quant)
                    quant = 0
                grouped.append(j)
                lastWord = j
                                
        lin.append(grouped)
        wordsPerSymbol.append(quant+1)
        compressionMatrix.append(wordsPerSymbol)
        wordsPerSymbol = []
        grouped = []
        quant = 0 
        lastWord = ''        
    return [lin,compressionMatrix]

def convert2nucleotidesWInterval(trace_set,n_samples=1000,step=[0]):    
    len_alphabet = len(step)  
    ms_unaligned = []
    nucleotides_seq = []
    for index,item in enumerate(trace_set):
        for j,k in enumerate(trace_set[index]):
            if j == n_samples :
                break
            nIndex = findPos(k,step)#round((trace_set[index][j]+127)/step)
            if nIndex > len_alphabet:
                nIndex = len_alphabet
            if nIndex < 0:
                nIndex = 0
            nucleotides_seq.append(nucleotide_alphabet[nIndex])
        ms_unaligned.append(nucleotides_seq)
        nucleotides_seq = []    
    return ms_unaligned    

def findPos(sample,steps):
    pos = 0
    for i in steps:
        if sample <= i:
            break
        pos += 1   
    return pos  
  
#All vectors here should have the same size
#This function returns max lenght in that column when vectors have gaps post MSA
def findMaxLenghtPerIndex(msa_zip,compressionRef):
    
    MaxLenghtPerIndex = [0] * len(msa_zip[0]) #columns with compression information
    n_traces = len(msa_zip)
    n_samples = len(msa_zip[0])
    #This might have to be fixed , before it was compressionRef[1]
    n_per_trace = [0] * len(compressionRef) #n_samples should be equal to this size    
    #Search every column for biggest ref based on compression reference matrix
    for i in range(0,n_samples):
        for j in range(0,n_traces):
            #print('MSA:' + msa_zip[j][i] + ' ' +  str(j) + ' ' + str(i))
            #print(len(n_per_trace))
            if msa_zip[j][i] != '-':                
                currentIndex = n_per_trace[j]
                n_per_trace[j] += 1 # this number was used, so increase the reference.
                #print(MaxLenghtPerIndex[i])
                #print(compressionRef[j][currentIndex])
                if MaxLenghtPerIndex[i] < compressionRef[j][currentIndex]:
                    MaxLenghtPerIndex[i] = compressionRef[j][currentIndex]      
                    
    return MaxLenghtPerIndex

def findMaxLenghtPerIndexV2(msa_zip,compressionRef):
    
    MaxLenghtPerIndex = [0] * len(msa_zip[0]) #columns with compression information
    n_traces = len(msa_zip)
    n_samples = len(msa_zip[0])
    #This might have to be fixed , before it was compressionRef[1]
    n_per_trace = [0] * len(compressionRef) #n_samples should be equal to this size    
    #Search every column for biggest ref based on compression reference matrix
    for i in range(0,n_samples):
        for j in range(0,n_traces):
            #print('MSA:' + msa_zip[j][i] + ' ' +  str(j) + ' ' + str(i))
            #print(len(n_per_trace))
            if msa_zip[j][i] != '-':                
                currentIndex = n_per_trace[j]
                n_per_trace[j] += 1 # this number was used, so increase the reference.
                #print(MaxLenghtPerIndex[i])
                #print(compressionRef[j][currentIndex])
                if MaxLenghtPerIndex[i] < compressionRef[j][currentIndex]:
                    MaxLenghtPerIndex[i] = compressionRef[j][currentIndex]      
                    
    return MaxLenghtPerIndex

def decompress_gaps(msa_zip,compressionRef):
    #create a function here with Max lenght MSA_s
    #print(compressionRef) 
    maxPerIndex = findMaxLenghtPerIndex(msa_zip,compressionRef)
    decompress_traces = []
    traces_Aligned = []
    #n_per_trace = [0] * len(compressionRef[0]) #n_samples should be equal to this size
    n = 0 
      
    for i in range(0,len(msa_zip)):        
        for j in  range(0,len(msa_zip[0])):
            #print('MSA: ' + msa_zip[i][j] + ' ' +  str(j) + ' ' + str(i))
            if msa_zip[i][j] != '-':
                refQuantity = compressionRef[i][n]
                decompress_traces = decompress_traces + ([msa_zip[i][j]] * refQuantity) + (['-']*(maxPerIndex[j]-refQuantity))
                n += 1 
            else:
                decompress_traces = decompress_traces +  (['-']*(maxPerIndex[j]))
        n = 0
        traces_Aligned.append(decompress_traces)
        decompress_traces = []
    
    
    return traces_Aligned

def decompress_gapsV2(msa_zip,compressionRef):
    #create a function here with Max lenght MSA_s
    #print(compressionRef) 
    maxPerIndex = findMaxLenghtPerIndex(msa_zip,compressionRef)
    decompress_traces = []
    traces_Aligned = []
    #n_per_trace = [0] * len(compressionRef[0]) #n_samples should be equal to this size
    n = 0 
      
    for i in range(0,len(msa_zip)):        
        for j in  range(0,len(msa_zip[0])):
            #print('MSA: ' + msa_zip[i][j] + ' ' +  str(j) + ' ' + str(i))
            if msa_zip[i][j] != '-':
                refQuantity = compressionRef[i][n]
                decompress_traces = decompress_traces + ([msa_zip[i][j]] * refQuantity) + (['-']*(maxPerIndex[j]-refQuantity))
                n += 1 
            else:
                decompress_traces =  decompress_traces +  (['-']*(maxPerIndex[j]))
        n = 0
        traces_Aligned.append(decompress_traces)
        decompress_traces = []
    
    
    return traces_Aligned

def decompress(refTraces,msa_zip,msa_unzip):
    #create a function here with Max lenght MSA_s
    decompress_traces = []
    traceAligned = []
    gap = 0
    pos = 0
    for i in range(0,len(msa_zip)):
        for symbol in  msa_zip[i]:
            if symbol == '-':
                traceAligned.append(gap) 
            else:
                while symbol == msa_unzip[i][pos] :
                    traceAligned.append(refTraces[i][pos])
                    pos += 1
        decompress_traces.append(traceAligned)
        pos = 0
        traceAligned = []
        
    return decompress_traces

def msaAlignment(trace_src,n_traces,window,alphabet_size,alignment_method=1,print_src=1,eliminate_gap=0,print_gap=0,mergeConsensus=0,mergeBack=0,compress=0): 
    #Parameters
    initial = window[0]
    n_samples = window[1]    
    # trace_src = '\good-misaligned1000samples.trs' #'\\rndDelays500traces.trs'#'\good-misaligned1000samples_CPAHere.trs'# '\good-misaligned1000samples_endtrim.trs' #
    file_trace = in_path + trace_src 
    file_inspector = out_path + '\\mafft_' + str(n_traces) + 't_' + str(n_samples) + 's_' + str(alphabet_size) + 'alf'
    input = fasta_in + '\preAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
    output = fasta_out + '\postAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
    base_trace = file_inspector + '_src'
    mafft = 'mafft-fftns --ep 0.123 ' + options +' --maxiterate 1000 ' + input  + ' > ' + output    
    traces = trsfile.open(file_trace, 'r')
    desired_traces = selectWindow(traces,initial,n_samples,n_traces)
    
    start = time.time()
    
    step = step_minmaxTraces(desired_traces,alphabet_size)
    
    #Y-axis
    if alignment_method == 1 or alignment_method == 5:
        ms_unaligned = convert2nucleotides(desired_traces,alphabet_size-1,n_samples,step)
        file_inspector += '_y'
    #DNA    
    elif alignment_method == 2:
        ms_unaligned = convert2dna(desired_traces,6,n_samples,42.33, dna_alphabet)
        file_inspector += '_yD'
    #SaX
    elif alignment_method == 3:
        sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal',alphabet=nucleotide_alphabet[:alphabet_size])
        ms_unaligned = sax.fit_transform(desired_traces)    
        file_inspector += '_SaX'
    # PAA transformation
    elif alignment_method == 4:
        window_size = 2
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        paa_trace = paa.transform(desired_traces)
        ms_unaligned = convert2nucleotides(paa_trace,alphabet_size-1,(n_samples/window_size),step-1)
        file_inspector += '_PAA'
    #Text
    else :
        file_inspector += '_txt'
        ms_unaligned = trace2text(desired_traces,n_samples)
        mafft = 'mafft --text ' + input + ' > ' + output
    
    if compress == 1:
       ms_unalignedC = compressWSymbols(ms_unaligned)#compressSequence(ms_unaligned)
       file_inspector += 'C'
       print_msa(ms_unalignedC[0],input,n_samples)      
    else:
       print_msa(ms_unaligned,input,n_samples)
        
    os.system(mafft)

    traces_with_gaps = readMSA(output)
    
    if compress == 1:
        #print(ms_unalignedC[1])
        unziped = decompress_gaps(traces_with_gaps,ms_unalignedC[1])
        #print(len(unziped))
        #print(len(unziped[0]))
        #print(len(desired_traces))
        traces_bioaligned = msa2trace(desired_traces,unziped)#decompress(desired_traces,traces_with_gaps,ms_unaligned)
    
    if alignment_method == 4:
        traces_bioaligned = paamsa2trace(desired_traces,traces_with_gaps,window_size)
    elif alignment_method == 5:
        traces_bioaligned = text2trace(traces_with_gaps,desired_traces)     
    else:
        traces_bioaligned = msa2trace(desired_traces,traces_with_gaps)
    
    end = time.time()

    total_time = end-start
    print("My running time in seconds was:\n"+str(total_time))
    consensus = []
    if print_gap == 1:
        if mergeConsensus == 1:
            consensus = findConsensus(traces_with_gaps)
            print_trs_consensus(traces,consensus,traces_bioaligned,file_inspector + '_gapConse', n_traces) # need a special treatment for the first trace , maybe create a new functions
            #(trace_src,consensus,to_print,file_name,total_traces): 
        else:
            print_trs(traces, traces_bioaligned, file_inspector + '_gap', n_traces) 
    #print(len(consensus + traces_bioaligned))     
    #eliminate_gap : 1 - Shift to original, 2 - avarege between gaps , 3 - avgCol
    if eliminate_gap == 1:        
        traces_bioaligned =  eliminateGaps(desired_traces,traces_with_gaps)  
        file_inspector += '_shiftOn'
    elif eliminate_gap == 2:
        for i in range(0, n_traces):
            traces_bioaligned[i] = averageMethod(traces_bioaligned[i])
        file_inspector += '_avg'
    elif eliminate_gap == 3 :
        traces_bioaligned = closeGapWAvarege(traces_bioaligned,traces_with_gaps)
        file_inspector += '_avgCol'

    if mergeBack == 1:
        traces_bioaligned = mergeTraceAt(traces,traces_bioaligned,initial,n_samples,n_traces)
        file_inspector += '_onSr'
        
    #file_inspector += '.trs'
    #Print final and Original section
    print_trs(traces,traces_bioaligned,file_inspector,n_traces)
    #print_trs_diff(traces,traces_bioaligned,file_inspector,n_traces)
    if print_src == 1:
        print_trs(traces,desired_traces,base_trace,n_traces)    
    
    return [consensus, traces_bioaligned, desired_traces]

def print_SectionAA(trace_src,n_traces,window,alphabet_size,alignment_method=1,compress=0):
        #Parameters
    initial = window[0]
    n_samples = window[1]    
    # trace_src = '\good-misaligned1000samples.trs' #'\\rndDelays500traces.trs'#'\good-misaligned1000samples_CPAHere.trs'# '\good-misaligned1000samples_endtrim.trs' #    
    file_trace = in_path + trace_src 
    file_inspector = out_path + '\\test_' + str(n_traces) + 't_' + str(n_samples) + 's_' + str(alphabet_size) + 'alf'
    input = fasta_in + '\preAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt    
    traces = trsfile.open(file_trace, 'r')
    desired_traces = selectWindow(traces,initial,n_samples,n_traces)    
    step = step_minmaxTraces(desired_traces,alphabet_size)    
    #Y-axis
    if alignment_method == 1 or alignment_method == 5:
        ms_unaligned = convert2nucleotides(desired_traces,alphabet_size-1,n_samples,step)
        file_inspector += '_y'
    #DNA    
    elif alignment_method == 2:
        ms_unaligned = convert2dna(desired_traces,6,n_samples,42.33, dna_alphabet)
        file_inspector += '_yD'
    #SaX
    elif alignment_method == 3:
        sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal',alphabet=nucleotide_alphabet[:alphabet_size])
        ms_unaligned = sax.fit_transform(desired_traces)    
        file_inspector += '_SaX'
    # PAA transformation
    elif alignment_method == 4:
        window_size = 2
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        paa_trace = paa.transform(desired_traces)
        ms_unaligned = convert2nucleotides(paa_trace,alphabet_size-1,(n_samples/window_size),step-1)
        file_inspector += '_PAA'
    #Text
    else :
        file_inspector += '_txt'
        ms_unaligned = trace2text(desired_traces,n_samples)        
    
    if compress == 1:
        ms_unalignedC = compressSequence(ms_unaligned)
        file_inspector += 'C'
        print_msa(ms_unalignedC,input,n_samples)
        return [desired_traces,ms_unalignedC]      
    else:
       print_msa(ms_unaligned,input,n_samples)
       return [desired_traces,ms_unaligned]

def msa_with_compression(trace_src,n_traces,window,alphabet_size,alignment_method=1,print_src=1,eliminate_gap=0,print_gap=0,mergeConsensus=0,mergeBack=0): 
    #Parameters
    initial = window[0]
    n_samples = window[1]    
    # trace_src = '\good-misaligned1000samples.trs' #'\\rndDelays500traces.trs'#'\good-misaligned1000samples_CPAHere.trs'# '\good-misaligned1000samples_endtrim.trs' #
    file_trace = in_path + trace_src 
    file_inspector = out_path + '\\test_' + str(n_traces) + 't_' + str(n_samples) + 's_' + str(alphabet_size) + 'alf'
    input = fasta_in + '\preAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
    output = fasta_out + '\postAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
    base_trace = file_inspector + '_src'
    mafft = 'mafft ' + options +' --maxiterate 1000 ' + input  + ' > ' + output    
    traces = trsfile.open(file_trace, 'r')
    desired_traces = selectWindow(traces,initial,n_samples,n_traces)
    
    start = time.time()
    
    step = step_minmaxTraces(desired_traces,alphabet_size)
    
    #Y-axis
    if alignment_method == 1 or alignment_method == 5:
        ms_unaligned = convert2nucleotides(desired_traces,alphabet_size-1,n_samples,step)
        file_inspector += '_y'
    #DNA    
    elif alignment_method == 2:
        ms_unaligned = convert2dna(desired_traces,6,n_samples,42.33, dna_alphabet)
        file_inspector += '_yD'
    #SaX
    elif alignment_method == 3:
        sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal',alphabet=nucleotide_alphabet[:alphabet_size])
        ms_unaligned = sax.fit_transform(desired_traces)    
        file_inspector += '_SaX'
    # PAA transformation
    elif alignment_method == 4:
        window_size = 2
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        paa_trace = paa.transform(desired_traces)
        ms_unaligned = convert2nucleotides(paa_trace,alphabet_size-1,(n_samples/window_size),step-1)
        file_inspector += '_PAA'
    #Text
    else :
        file_inspector += '_txt'
        ms_unaligned = trace2text(desired_traces,n_samples)
        mafft = 'mafft --text ' + input + ' > ' + output
    
    
    ms_unalignedC = compressSequence(ms_unaligned)
    file_inspector += 'C'
        #print_msa(ms_unalignedC,input,n_samples)      

    return ms_unalignedC
   
def trace_from_MSAFile(read_file,trace_src,n_traces,window,alphabet_size,alignment_method=1,print_src=1,eliminate_gap=0,print_gap=0,mergeConsensus=0,mergeBack=0,compress=0): 
    #Parameters
    initial = window[0]
    n_samples = window[1]    
    # trace_src = '\good-misaligned1000samples.trs' #'\\rndDelays500traces.trs'#'\good-misaligned1000samples_CPAHere.trs'# '\good-misaligned1000samples_endtrim.trs' #
    file_trace = out_path + trace_src 
    file_inspector = out_path + '\\Present_' + str(n_traces) + 't_' + str(n_samples) + 's_' + str(alphabet_size) + 'alf'
    output = fasta_out + read_file
    base_trace = file_inspector + '_src'      
    traces = trsfile.open(file_trace, 'r')
    desired_traces = selectWindow(traces,initial,n_samples,n_traces)    
    step = step_minmaxTraces(desired_traces,alphabet_size) 
    
    traces_with_gaps = readMSA(output)
       
    if compress == 1:
    #Y-axis
        if alignment_method == 1 or alignment_method == 5:
            ms_unaligned = convert2nucleotides(desired_traces,alphabet_size-1,n_samples,step)
            file_inspector += '_y'
        #DNA    
        elif alignment_method == 2:
            ms_unaligned = convert2dna(desired_traces,6,n_samples,42.33, dna_alphabet)
            file_inspector += '_yD'
        #SaX
        elif alignment_method == 3:
            sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal',alphabet=nucleotide_alphabet[:alphabet_size])
            ms_unaligned = sax.fit_transform(desired_traces)    
            file_inspector += '_SaX'
        # PAA transformation
        elif alignment_method == 4:
            window_size = 2
            paa = PiecewiseAggregateApproximation(window_size=window_size)
            paa_trace = paa.transform(desired_traces)
            ms_unaligned = convert2nucleotides(paa_trace,alphabet_size-1,(n_samples/window_size),step-1)
            file_inspector += '_PAA'
        #Text
        else :
            file_inspector += '_txt'
            ms_unaligned = trace2text(desired_traces,n_samples)
            
        traces_bioaligned = decompress(desired_traces,traces_with_gaps,ms_unaligned)
    
    if alignment_method == 4:
        traces_bioaligned = paamsa2trace(desired_traces,traces_with_gaps,window_size)
    elif alignment_method == 5:
        traces_bioaligned = text2trace(traces_with_gaps,desired_traces)     
    else:
        traces_bioaligned = msa2trace(desired_traces,traces_with_gaps)    

    consensus = []
    if print_gap == 1:
        if mergeConsensus == 1:
            consensus = findConsensus(traces_with_gaps)
            print_trs_consensus(traces,consensus,traces_bioaligned,file_inspector + '_gapConse', n_traces) # need a special treatment for the first trace , maybe create a new functions
        else:
            print_trs(traces, traces_bioaligned, file_inspector + '_gap', n_traces) 
    #print(len(consensus + traces_bioaligned))     
    #eliminate_gap : 1 - Shift to original, 2 - avarege between gaps , 3 - avgCol
    if eliminate_gap == 1:        
        traces_bioaligned =  eliminateGaps(desired_traces,traces_with_gaps)  
        file_inspector += '_shiftOn'
    elif eliminate_gap == 2:
        for i in range(0, n_traces):
            traces_bioaligned[i] = averageMethod(traces_bioaligned[i])
        file_inspector += '_avg'
    elif eliminate_gap == 3 :
        traces_bioaligned = closeGapWAvarege(traces_bioaligned,traces_with_gaps)
        file_inspector += '_avgCol'

    if mergeBack == 1:
        traces_bioaligned = mergeTraceAt(traces,traces_bioaligned,initial,n_samples,n_traces)
        file_inspector += '_onSr'    
    
    #file_inspector += '.trs'
    #Print final and Original section
    print_trs(traces,traces_bioaligned,file_inspector,n_traces)
    #print_trs_diff(traces,traces_bioaligned,file_inspector,n_traces)
    if print_src == 1:
        print_trs(traces,desired_traces,base_trace,n_traces)    
    
    return [consensus, traces_bioaligned, desired_traces]

def plot_trs_intervals(n_samples,stap,desired_traces,n = 1):
    n_timestamps = n_samples
    X = desired_traces
    plt.figure(figsize=(6, 4))
    plt.plot(X[0][:n_timestamps], 'o--', label='Trace 1')
    plt.plot(X[10][:n_timestamps], 'o--', label='Trace 2')
    #if n > 1:
     #   for i in range(1,n-1):
      #      plt.plot(X[n][:n_timestamps], 'o--', label='Trace '+str(i+1))
    #plt.plot(X[1][:n_timestamps], 'o--', label='Trace 2')
    plt.hlines(stap, 0, n_timestamps, color='g', linestyles='--', linewidth=1)    
    #sax_legend = mlines.Line2D([], [], color='#ff7f0e', marker='*',
    #                        label='SAX - {0} bins'.format(3))
    #first_legend = plt.legend(handles=[sax_legend], fontsize=8, loc=(0.76, 0.86))
    #ax = plt.gca().add_artist(first_legend)
    plt.legend(loc=(0.81, 0.93), fontsize=8)
    plt.xlabel('Time', fontsize=14)
    plt.title('Traces', fontsize=16)
    plt.show()
    
def plot_trs_avgVert(n_samples,stap,desired_traces):
    n_timestamps = len(desired_traces[0])
    X = desired_traces
    plt.figure(figsize=(6, 4))
    plt.plot(X[0][:n_timestamps], 'o--', label='Avg Distribution')
    plt.vlines(x=stap, ymin=([0]*len(stap)), ymax=([205]*len(stap)), colors='teal', ls='--', lw=2, label='Symbols division')
    plt.legend(loc=(0.81, 0.93), fontsize=8)
    plt.xlabel('Time', fontsize=14)
    plt.title('Samples Distribution (All Traces)', fontsize=16)
    plt.show()
    
def plotTable():
    x = [*range(20,250,20)]
    y1 = [0,0,1,1,1,1,1,1,1,1,1,1]
    y2 = [0,2,2,3,4,5,5,6,6,7,8,8]
    y3 = [0,0,0,3,6,8,8,8,8,8,8,8]
    plt.plot(x, y1,'o--', label = "Static")
    plt.plot(x, y2,'o--', label = "Elastic")    
    plt.plot(x, y3, 'o--', label = "MSA")
    plt.xlabel('Number of Traces')
    plt.ylabel('Sub-keys Found')
    plt.title('Found sub-keys per Traces')
    plt.legend()
    plt.show()

def plotTime():
    x = [*range(20,250,20)]
    y1 = [0,0,1,1,1,1,1,1,1,1,1,1]
    y2 = [0,2,2,3,4,5,5,6,6,7,8,8]
    y3 = [0,0,0,3,6,8,8,8,8,8,8,8]
    plt.plot(x, y1,'o--', label = "Static")
    plt.plot(x, y2,'o--', label = "Elastic")    
    plt.plot(x, y3, 'o--', label = "MSA")
    plt.xlabel('Number of Traces')
    plt.ylabel('Sub-keys Found')
    plt.title('Found sub-keys per Traces')
    plt.legend()
    plt.show() 


    
    
def alignWCompression():
    
    n_traces = 2
    initial = 0#7870#5070#4500#0#2000#361203#24000
    n_samples = 1999#3770#14800#5500 
    alphabet_size = 9
    maxSamples = 100
    #alignment_method : 1- y_axis & Nucleotides , 2- Dna 7 symbols, 3- Sax, 4- Paa , 5- Text
    alignment_method = 1
    #eliminate_gap : 1 - Shift to original, 2 - avarege between gaps , 3 - avgCol 
    connect = 1 
    print_gapTrs = 1
    print_src =1
    mergeBack = 0 
    compress = 0 
    mergeConsensusTrace = 0
    eliminate_gap = 0
    mergeConsensusTrace = 0
    print_gap = 0
    mergeConsensus = 0

    #trace_src =  '\DESenc_rndDelays500traces_Solution_endtrim.trs'  
    trace_src = '\good-misaligned1000samplesTEST2.trs'#'\good-misaligned1000samples.trs' #'\\rndDelays500traces.trs'#'\good-misaligned1000samples_CPAHere.trs'# '\good-misaligned1000samples_endtrim.trs' #
    file_trace = in_path + trace_src 
    file_inspector = out_path + '\\0A_' + str(n_traces) + 't_' + str(n_samples) + 's_' + str(alphabet_size) + 'alf'
    input = fasta_in + '\preAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
    output = fasta_out + '\postAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
    base_trace = file_inspector + '_src'
    #mafft = 'mafft ' + options +' --maxiterate 1000 ' + input  + ' > ' + output    
    mafft = 'mafft ' + options  + input  + ' > ' + output    
    traces = trsfile.open(file_trace, 'r')
    desired_traces = selectWindow(traces,initial,n_samples,n_traces)

    start = time.time()
    step = step_minmaxTraces(desired_traces,alphabet_size)

    #Y-axis
    if alignment_method == 1 or alignment_method == 5:
        stap = [-75,-50,-25,0,10,30,80]#trac[-25,40] [80] was the best result
        ms_unaligned = convert2nucleotidesWInterval(desired_traces,n_samples,stap) 
        #ms_unaligned = convert2nucleotides(desired_traces,alphabet_size-1,n_samples,step)
        #convert2nucleotidesWInterval(trace_set,n_samples=1000,step=[])#
        file_inspector += '_y'
    #DNA    
    elif alignment_method == 2:
        ms_unaligned = convert2dna(desired_traces,6,n_samples,42.33, dna_alphabet)
        file_inspector += '_yD'
    #SaX
    elif alignment_method == 3:
        sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal',alphabet=nucleotide_alphabet[:alphabet_size])
        ms_unaligned = sax.fit_transform(desired_traces)    
        file_inspector += '_SaX'
    # PAA transformation
    elif alignment_method == 4:
        window_size = 2
        paa = PiecewiseAggregateApproximation(window_size=window_size)
        paa_trace = paa.transform(desired_traces)
        ms_unaligned = convert2nucleotides(paa_trace,alphabet_size-1,(n_samples/window_size),step-1)
        file_inspector += '_PAA'
    #Text
    else :
        file_inspector += '_txt'
        ms_unaligned = trace2text(desired_traces,n_samples)
        mafft = 'mafft ' + input + ' > ' + output

    if compress == 1:
        ms_unalignedC = compressWSymbolsMax(ms_unaligned,maxSamples)#compressSequence(ms_unaligned)
        file_inspector += 'C'
        print_msa(ms_unalignedC[0],input,n_samples)      
    else:
        print_msa(ms_unaligned,input,n_samples)
        
    os.system(mafft)

    traces_with_gaps = readMSA(output)

    if compress == 1:
        #print(ms_unalignedC[1])
        unziped = decompress_gapsV2(traces_with_gaps,ms_unalignedC[1])#decompress_gaps(traces_with_gaps,ms_unalignedC[1])#[1])
        traces_bioaligned = msa2tracev2(desired_traces,unziped)#decompress(desired_traces,traces_with_gaps,ms_unaligned)
    else:
        if alignment_method == 4:
            traces_bioaligned = paamsa2trace(desired_traces,traces_with_gaps,window_size)
        elif alignment_method == 5:
            traces_bioaligned = text2trace(traces_with_gaps,desired_traces)     
        else:
            traces_bioaligned = msa2trace(desired_traces,traces_with_gaps)

    end = time.time()

    total_time = end-start
    print("My running time in seconds was:\n"+str(total_time))
    consensus = []
    if print_gap == 1:
        if mergeConsensus == 1:
            consensus = findConsensus(traces_with_gaps)
            print_trs_consensus(traces,consensus,traces_bioaligned,file_inspector + '_gapConse', n_traces) # need a special treatment for the first trace , maybe create a new functions
            #(trace_src,consensus,to_print,file_name,total_traces): 
        else:
            print_trs(traces, traces_bioaligned, file_inspector + '_gap', n_traces) 
    print(len(consensus + traces_bioaligned))     
    #eliminate_gap : 1 - Shift to original, 2 - avarege between gaps , 3 - avgCol
    if eliminate_gap == 1:        
        traces_bioaligned =  eliminateGaps(desired_traces,traces_with_gaps)  
        file_inspector += '_shiftOn'
    elif eliminate_gap == 2:
        for i in range(0, n_traces):
            traces_bioaligned[i] = averageMethod(traces_bioaligned[i])
        file_inspector += '_avg'
    elif eliminate_gap == 3 :
        traces_bioaligned = closeGapWAvarege(traces_bioaligned,traces_with_gaps)
        file_inspector += '_avgCol'

    if mergeBack == 1:
        traces_bioaligned = mergeTraceAt(traces,traces_bioaligned,initial,n_samples,n_traces)
        file_inspector += '_onSr'
        
    #file_inspector += '.trs'
    #Print final and Original section
    print_trs(traces,traces_bioaligned,file_inspector,n_traces)
    #print_trs_diff(traces,traces_bioaligned,file_inspector,n_traces)
    if print_src == 1:
        print_trs(traces,desired_traces,base_trace,n_traces)         
    return 

def consensus_Avg(consensus,window):   
    avgWindow = []         
    for i in range(0,len(consensus)):
        avgWindow.append(sum(consensus[i:i+window])/window)    
    return avgWindow

def counterMeasurePrediction(consensus,window,threshold):
    isCounterMeasure = []    
    avg = consensus_Avg(consensus,window)
    for i in avg:
        if i >= threshold:
            isCounterMeasure.append(127)
        else:
            isCounterMeasure.append(-126)            
    return isCounterMeasure

def eliminateGaps_by_Counter(traces_gaps,refTraces,counterPred):
    traceFromMsa = []
    traceAligned = []
    n = 0 # points to previous value of trace
    gap = 0#np.nan    
    for index, seqAA in enumerate(traces_gaps) :
        n = 0            
        for j in range(0,len(seqAA)) :            
            if seqAA[j] == '-' and counterPred[j] != 127:                
                traceAligned.append(gap)                
            else:
                traceAligned.append(refTraces[index][n])
                n += 1                
                if n >= len(refTraces[0]):
                    break
                #print('My last N : ' + str(n))               
        traceFromMsa.append(traceAligned)
        traceAligned = []            
    #It can mess up the convertion back!                  
    return traceFromMsa

def findNonCounter(counterPred):
    notCounter = []
    begin = 0
    end = 0    
    for i in range(1,len(counterPred)):
        if counterPred[i] == 127  and counterPred[i-1] == -126:
            begin = i  
        if counterPred[i] == -126  and counterPred[i-1] == 127:
            end = i
            notCounter.append([begin,end])           
    return notCounter
    
def keep_size(traces):
    shorty = []
    min = len(traces[0])
    for i in traces:
        if len(i) < min:
            min = len(i)
    for i in traces:
        shorty.append(i[:min])      
    return shorty

#New functions need more testing -------------22/11
def consensus_As_freq(consensus,n_traces):   
    freqWindow = []    
    for i in range(0,len(consensus)):
        freqWindow.append(round((n_traces - i)/n_traces))      
    return freqWindow

def consensus_Avg_middle(consensus,window):   
    avgWindow = []    
    for i in range(window/2,len(consensus)-(window/2)):
        avgWindow.append(sum(consensus[i-(window/2):i+window])/window)    
    return avgWindow

#[lin,compressionMatrix]
def fetchFromCompression(ms_granular,compressionMatrix):
    subSequences = []
    decompressed = []
    decomp_pos = [0]*len(compressionMatrix) 
    dmin = minVec(compressionMatrix)
    for j in range(0,dmin):#len(compressionMatrix[0])):        
        for i in range(0,len(compressionMatrix)):
            #print("To em :" + str(i) + " " + str(j))
            decompressed.append(ms_granular[i][decomp_pos[i]:decomp_pos[i]+compressionMatrix[i][j]])
            decomp_pos[i] += compressionMatrix[i][j]
        subSequences.append(decompressed)
        decompressed = []
    return subSequences

def concate_Granular(granular):
    concated = copy.copy(granular[0])
    for i in range(1,len(granular)):
        for row in range(0,len(granular[i])):
            #print(str(i) +  ' ' + str(row))
            concated[row] = concated[row] + granular[i][row]
    return concated

def minVec(vec):
    min = len(vec[0])
    for i in vec:
        if min > len(i):
            min = len(i)
    return min
    
def decTobin(dec):
    n = dec
    strg = ''
    while(n != 0):
        strg = str(n % 2) + strg
        n = int(n / 2)       
        
    return strg

def readCPA():
    maxCoeff8bytes = []
    finalCoefs = []
    trace_src = ['MAFFT_100t_6000s_7alph_src_staticAligned','MAFFT_100t_6000s_7alph_src_cpaGapAligned',
                 'MAFFT_100t_6000s_7alph_src_cpaGapShiftAligned','MAFFT_100t_6000s_7alph_src_cpaNotAligned']
    for i in trace_src:
        traces = trsfile.open('Data/result_traces/' + i + '.trs', 'r')
        for trace in traces:
            a = min(trace)
            b = max(trace)
            if abs(a) > b:
                maxCoeff8bytes.append(abs(a))
            else:
                maxCoeff8bytes.append(b)
        finalCoefs.append(maxCoeff8bytes[8:])
        maxCoeff8bytes = []   
    
    return finalCoefs

def plotTableCPAwithoutCounter():
    x = [*range(0,8)]
    y = readCPA() 
    plt.plot(x, y[0],'o--', label = "Static alignment")
    plt.plot(x, y[1],'o--', label = "Gaps to zero")#"MSA-based alignment")    
    plt.plot(x, y[2], 'o--', label = "Static by MSA")#"MSA-based static-like alignment")
    plt.plot(x, y[3], 'o--', label = "No alignment")
    plt.xlabel('Output byte',fontsize=25)
    plt.ylabel('Highest CPA coefficient per output byte',fontsize=25)
    plt.rc('legend', fontsize=18)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.legend(fontsize=22)
    #plt.title('CPA comparison per alignment method')
    plt.legend()
    plt.show()
    

def print_corr_heat(file_a,title):
    
    traces_result = trsfile.open(file_a, 'r')    
    #staticAlign = trsfile.open('noisy2000tracesRegion + StaticAlign.trs', 'r')  #    
    corr_orig = np.corrcoef(traces_result)
    corr_final = np.corrcoef(traces_result)
    #corr_static = np.corrcoef(staticAlign)
    fig = plt.figure()
    N_rows_a, _ = corr_orig.shape
    N_rows_b, _ = corr_final.shape
    grid=gs.GridSpec(1,1)#, height_ratios=[N_rows_a,N_rows_b], width_ratios=[50,1])
    ax1 = fig.add_subplot(grid[0,0])
    #ax2 = fig.add_subplot(grid[1,0], sharex=ax1)
    #ax3 = fig.add_subplot(grid[2,0], sharex=ax1)
    #cax = fig.add_subplot(grid[:,1])

    ax1.set_title(title)
    #ax2.set_title('Pairwise correlation : MAFFT MSA Aligned Traces ')
    #ax2.set_title('Pairwise correlation : Static Aligned Traces')

    mask = np.triu(corr_orig)
    mask1 = np.triu(corr_final)
   # mask2 = np.triu(corr_static)          

    #fig, (ax1, ax2) = plt.subplots(ncols=2)
    #anx = sns.heatmap(corr_orig)
    #anx1 = sns.heatmap(corr_final)
    sns.heatmap(corr_orig,vmin=0.6,annot=True, cmap="inferno",mask=mask, ax=ax1)#, cbar_ax=cax,linewidths=.5)
    #sns.heatmap(corr_final,vmin=0.6,annot=True, cmap="inferno",mask=mask1, ax=ax2, cbar_ax=cax,linewidths=.5)
    #sns.heatmap(corr_static,vmin=0.6,annot=True, cmap="inferno",mask=mask2, ax=ax2, cbar_ax=cax,linewidths=.5)
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.show()
    return 0


def print_corr_heat_alphabet2():
    
    trace_src = ['MAFFT_100t_6000s_7alph_src','14_01GLobP10t_5998s_3a_y_Counter',
                '14_01GLobP10t_5998s_5a_y_Counter','14_01GLobP10t_5998s_7a_yD_Counter','14_01GLobP10t_5998s_8a_y_Counter',
                '14_01GLobP10t_5998s_15a_y_Counter','14_01GLobP10t_5998s_20a_y_Counter']
    trace_title = ['Source set','MSA : 2 Alphabet size',
                'MSA: 5 Alphabet size','MSA: 7 Alphabet size', 'MSA (Protein): 7 Alphabet size',
                'MSA: 15 Alphabet size','MSA: 20 Alphabet size']
    corr_total = []
    n = 0
    for i in trace_src:
        #traces_result = trsfile.open('Data/result_traces/' + i + '.trs', 'r')         
        print_corr_heat('Data/result_traces/' + i + '.trs',trace_title[n])
        n = n + 1

def print_corr_heat_alphabet3():
    
    trace_src = ['MAFFT_100t_6000s_7alph_src']
    #trace_title = ['Source set']
    for i in range(2,21):
        trace_src = trace_src + ['17_01GLobP10t_5998s_' + str(i) + 'a_y_gap']
        #trace_title = trace_title + ['MSA:' + str(i) + 'Alphabet size']
    corr_total = []
    n = 0
    for i in trace_src:
        #traces_result = trsfile.open('Data/result_traces/' + i + '.trs', 'r')
        corr_total.append(find_MinMaxAvg_Corr('Data/result_traces/' + trace_src[n] + '.trs'))
        #print_corr_heat('Data/result_traces/' + i + '.trs',trace_title[n])
        n = n + 1
    return corr_total

def plotMinMaxAvg(corr_total):
    maximum = []
    minimum = []
    avgs    = []
    for i in corr_total:
        avgs.append(i[0])
        maximum.append(i[1])
        minimum.append(i[2])
        
    x = ['Source']#[str(*range(2,21))]
    for i in range(2,21):
        x = x + [str(i)]    
    plt.plot(x, maximum,'o--', label = "Maximum")
    plt.plot(x, minimum,'o--', label = "Minimum")    
    plt.plot(x, avgs, 'o--', label = "Average")   
    plt.xlabel('Alphabet size')
    plt.ylabel('Correlation Coefficent')
    plt.title('Alphabet lenght : Pairwise correlation')
    plt.legend()
    plt.show()               

def find_MinMaxAvg_Corr(file_a):
    traces_result = trsfile.open(file_a, 'r') 
    corr_orig = np.corrcoef(traces_result)
    sum = 0
    bigger = -1
    smaller = -1
    current = corr_orig[0][0]    
    for i in range(0,len(corr_orig)):
        for j in range(0,len(corr_orig[0])):
            current = corr_orig[i][j]
            if i != j:
                sum = sum + current
                if bigger < current:
                    bigger = current
                if smaller > current:
                    smaller = current
    avg = sum/((len(corr_orig[0])-1)*(len(corr_orig[0])))
    return [avg,bigger,smaller]
                    
                
def plotFoundKeys():
    x = [*range(20,250,20)] + [250]
    y = [[0,0,1,1,1,1,1,1,1,1,1,1,1],
         [0,0,1,3,3,3,4,4,4,6,6,7,8],
         [0,0,0,1,4,6,8,8,8,8,8,8,8]]
    plt.plot(x, y[0],'o--', label = "Static alignment")
    plt.plot(x, y[1],'o--', label = "Elastic alignment")    
    plt.plot(x, y[2], 'o--', label = "MSA-based alignment")
    plt.legend(fontsize=22)
    plt.xlabel('Number of traces',fontsize=25)
    plt.ylabel('Number of sub-keys found',fontsize=25)
    plt.rc('legend', fontsize=18)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #plt.title('Found sub-keys per traces')
    plt.legend()
    plt.savefig('CPAcomparisonWithout_counter.pdf')  
    plt.show()  
            
            
#print_corr_heat_alphabet2()