from msaAuxFunctions import *
exec(open("msaAuxFunctions.py").read())

#Linux 
#nucleotide_alphabet =  ['C','S','T','A','G','P','D','E','Q','N','H','R','K','M','I','L','V','W','Y','F']
nucleotide_alphabet = ['Y','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','V','W','A']
dna_alphabet = ['A','M','C','S','G','K','T']
in_path = 'Data/source_traces'
out_path = 'Data/result_traces'
fasta_in = 'Data/fasta_in'
fasta_out = 'Data/fasta_out'
end_txt = '.txt'
options = ' --large --globalpair  '#' --localpair '#'--globalpair ' #'--localpair ' #

#-------------------------------------------------------------------------

#def msaAlignment(trace_src,n_traces,window,alphabet_size,alignment_method=1,print_src=1,eliminate_gap=0,print_gap=0,mergeConsensus=0,mergeBack=0,compress=0): 
#Parameters
n_traces = 60
initial = 0#7870#5070#4500#0#2000#361203#24000
n_samples = 8400#1999#3770#14800#5500 
alphabet_size = 8
maxSamples = 100
divs = [30,50,80,100,140,165]#[-75,-50,-25,0,10,30,80]#trac[-25,40] [80] was the best result
stap = [num - 126 for num in divs  ]#[range(-126,127,12)]#
counter_window = 100
counter_threshold = 4
#alignment_method : 1- y_axis & Nucleotides , 2- Dna 7 symbols, 3- Sax, 4- Paa , 5- Text
alignment_method = 2
#eliminate_gap : 1 - Shift to original, 2 - avarege between gaps , 3 - avgCol 
connect = 1 
print_gapTrs = 1
print_src =1
mergeBack = 0 
compress = 0 
mergeConsensusTrace = 0
eliminate_gap = 2
mergeConsensusTrace = 2
print_gap = 1
mergeConsensus = 2
consensus_as_frequency = 0

trace_src = '/powerTrace.trs'
file_trace = in_path + trace_src 
file_inspector = out_path + '/OutputAlignment' + str(n_traces) + 't_' + str(n_samples) + 's_' + str(alphabet_size) + 'a'
input = fasta_in + '/preAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
output = fasta_out + '/postAlign' + str(n_traces) + 't_' + str(n_samples) + 's' + end_txt
base_trace = file_inspector + '_src'
#mafft = 'mafft ' + options +' --maxiterate 1000 ' + input  + ' > ' + output    
mafft = 'wsl.exe mafft --globalpair ' + input  + ' > ' + output 
traces = trsfile.open(file_trace, 'r')
desired_traces = selectWindow(traces,initial,n_samples,n_traces)

start = time.time()

step = step_minmaxTraces(desired_traces,alphabet_size)


#Y-axis
if alignment_method == 1 or alignment_method == 5:
    ms_unaligned = convert2nucleotides(desired_traces,alphabet_size-1,n_samples,step)
    file_inspector += '_y'
elif alignment_method == 6:    
    ms_unaligned = convert2nucleotidesWInterval(desired_traces,n_samples,stap) 
    file_inspector += '_yStap'
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
    ms_unalignedC = compressWSymbolsMax(ms_unaligned,maxSamples)#
    file_inspector += 'C'
    print_msa(ms_unalignedC[0],input,n_samples)      
else:
    print_msa(ms_unaligned,input,n_samples)
    
os.system(mafft)

traces_with_gaps = readMSA(output)

if compress == 1:
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
    if mergeConsensus == 2:
        consensus = findConsensus(traces_with_gaps)
        print_trs_counterFind(traces,consensus,traces_bioaligned,file_inspector + '_Counter', n_traces,counter_window,counter_threshold)
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

isCounter = counterMeasurePrediction(consensus,counter_window,counter_threshold)  
zonder_gaps = eliminateGaps_by_Counter(traces_with_gaps,desired_traces,isCounter)
without = keep_size(zonder_gaps)

print_trs(traces,without,file_inspector+'noGap',n_traces)

#trace_avg = '\DES_Rnd_TestSet_MicroZone_Distribution_Avg.trs'
#file_traceavg = in_path + trace_avg
#tracesavg = trsfile.open(file_traceavg, 'r')
#avg_t = selectWindow(tracesavg,0,211,1)
#divs = [30,50,80,100,140,165]
#plot_trs_avgVert(211,divs,avg_t)

#plot_trs_intervals(n_samples,stap,desired_traces)

#exec(open("ExampleCode.py").read())

#import cProfile

#cProfile.run("align()")