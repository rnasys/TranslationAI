########################################################################################
'''This program takes as input the .fa file and outputs a .h5 file with datapoints
   of the form (X, Y), which can be understood by Keras models.'''
# Usage: fa_to_h5_converter(fnIn,CHUNK_SIZE, fnOut)
#       sys.argv[1], input standard .fa file path
#       sys.argv[2], output .h5 file path
########################################################################################

import h5py
import sys
import time
from translationai.utils import *

if __name__ == '__main__':
    start_time = time.time()

    fnIn=sys.argv[1]
    fnOut=sys.argv[2]

    CHUNK_SIZE=1
    pad_len = CL_max // 2
    SEQ = [] # string, e.g. 'ATGC'
    STRAND = [] # string, e.g. '+'
    TX_START = [] # string of an integer, e.g. '12445'
    TX_END = []  # string of an integer, e.g. '12445'
    TIS = [] # string of integers , e.g. ['12445,13454,']
    TTS = [] # string of integers , e.g. ['12445,13454,']

    print('----Reading in .fa raw data ...----')
    line_count = 0
    with open(fnIn, 'r') as file:
        for line in file:
            if not line.strip(): # empty line
                continue
            if line.startswith('>'):
                words=line.strip('\n').split(':')
                STRAND.append('+')
                try:
                    words2=re.split('\(|\)',words[1])
                except:
                    raise Exception('Seq header format error: %s'%line)
                if len(words2)!=5 and len(words2)!=7:
                    print(line)
                    raise Exception('.fa header format error!')
                [TIS_pos,TTS_pos]=words2[-2].split(' ')
                TIS.append([TIS_pos])
                TTS.append([TTS_pos])
            else:
                seq_content=line.strip().upper()
                if not re.match(r'^[ACGTN]+$', seq_content):
                    raise ValueError("Invalid sequence. Only A, C, G, T, and N are allowed.")
                seq_len=len(seq_content)
                TX_START.append('0')
                TX_END.append(str(seq_len-1))
                seq_temp = 'N'*(pad_len) + seq_content + 'N'*(pad_len)
                SEQ.append(seq_temp)

    print('----Creating .h5 dataset for translationAI ...----')
    print('    *Sequences in one block: '+str(CHUNK_SIZE)+'. Block number: '+str(len(SEQ)//CHUNK_SIZE)+'.')
    SEQ=np.asarray(SEQ)
    h5f2 = h5py.File(fnOut, 'w')
    for i in range(SEQ.shape[0]//CHUNK_SIZE):
        # Each block in the dataset has CHUNK_SIZE genes
        if (i+1) == SEQ.shape[0]//CHUNK_SIZE:
            NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0]%CHUNK_SIZE
        else:
            NEW_CHUNK_SIZE = CHUNK_SIZE
        X_batch = []
        Y_batch = [[] for t in range(1)]
        for j in range(NEW_CHUNK_SIZE):
            idx = i*CHUNK_SIZE + j
            X, Y = create_datapoints(SEQ[idx], STRAND[idx],TX_START[idx], TX_END[idx], TIS[idx], TTS[idx])
            X_batch.extend(X)
            for t in range(1):
                Y_batch[t].extend(Y[t])
        X_batch = np.asarray(X_batch).astype('int8')
        for t in range(1):
            Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')
        h5f2.create_dataset('X' + str(i), data=X_batch)
        h5f2.create_dataset('Y' + str(i), data=Y_batch)
    h5f2.close()

    print("----File conversion done (%s to %s)! Time used: %d seconds.----" %
          (fnIn.split('/')[-1], fnOut.split('/')[-1], time.time() - start_time))
