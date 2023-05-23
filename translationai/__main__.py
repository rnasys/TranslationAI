########################################################################################
'''This program utilizes translationAI to predict potential Translation Initiation Sites
   (TISs), Translation Termination Sites (TTSs), and Open Reading Frames (ORFs) within
   query sequences in .fa input file.'''
########################################################################################

import sys
import time
import os
import h5py
from keras.models import load_model
from translationai.utils import *
import argparse
import logging
from pkg_resources import resource_filename

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # INFO, WARNING, and ERROR messages are not printed

def get_options():
    welcome_message = 'Welcome to translationAI ({})'.format(time.ctime())
    print("{:-^100}".format(welcome_message))
    parser = argparse.ArgumentParser(description='translationai (Version: 1.0) is utilized to predict '
                'potential Translation Initiation Sites (TISs), Translation Termination Sites (TTSs), and Open'
                ' Reading Frames (ORFs) within query sequences in .fa input file.')
    parser.add_argument('-I', metavar='input', required=True, help='Path to the input .fa query file.'
                        ' e.g. ./example_data/refSeq_hg19_random_100.fa')
    parser.add_argument('-t', metavar='thresholds', required=True,
                        help='Threshold k values for TIS/TTS prediction. If k>1, output the top-k TISs'
                             ' and TTSs in each sequence; if k<1, output the score>=k TISs and TTSs.'
                             ' e.g. 0.9,0.9')
    args = parser.parse_args()
    return args

def TIS_TTS_predictor(modelScale, modelsUsed,TIS_score_cutoff, TTS_score_cutoff,input_fa_fn,TIS_TTS,use_pad, fnOut1, fnOut2):

    start_time = time.time()

    print("{:-^100}".format('Reading in translationAI models'))
    BATCH_SIZE = 6
    version = modelsUsed.split(',')
    model = [[] for v in range(len(version))]
    for v in range(len(version)):
        modelNA = 'models/translationAI_' + modelScale + '_' + version[v] + '.h5'
        model[v] = load_model(resource_filename(__name__, modelNA))
        model[v].compile(loss=categorical_crossentropy_2d, optimizer='adam')

    h5f_name = input_fa_fn[:-3] + '.h5'
    if not os.path.exists(h5f_name):
        print("{:-^100}".format('Creating .h5 dataset from .fa file'))
        prog_path = resource_filename(__name__, 'fa_to_h5_converter.py')
        command = 'python ' + prog_path + ' ' + input_fa_fn + ' ' + input_fa_fn[:-3] + '.h5'
        os.system(command)
    else:
        print("{:-^100}".format('Reading in input .h5 file'))
    h5f = h5py.File(h5f_name, 'r')
    num_idx = len(h5f.keys()) // 2
    seqIn = open(input_fa_fn, 'r').readlines()
    seqNum = len(seqIn) // 2
    if num_idx != seqNum:
        print([num_idx, seqNum])
        raise Exception('!!!ERROR: The sequence numbers from the .h5 file and the .fa file do not match!')

    print("{:-^100}".format('Predicting TISs and TTSs'))
    fhOut1 = open(fnOut1, 'w')
    fhOut2 = open(fnOut2, 'w')
    for idx in range(num_idx):
        if not (idx%10):
            print('    Procssing the %d/%d sequence ...'%(idx+1, num_idx))
        Y_pred_TIS = [[] for t in range(1)]
        Y_pred_TTS = [[] for t in range(1)]
        X = h5f['X' + str(idx)][:]
        Y = h5f['Y' + str(idx)][:]
        Xc, Yc = clip_datapoints(X, Y, int(modelScale), 1)  # set NGPU=1
        Yps = [np.zeros(Yc[0].shape)]
        for v in range(len(version)):
            Yp = model[v].predict(Xc, batch_size=BATCH_SIZE)
            if not isinstance(Yp, list):
                Yp = [Yp]
            Yps[0] += Yp[0] / len(version) # mean of the ensemble predictions is used
        seq_len = len(seqIn[idx * 2 + 1])
        is_expr = (Yc[0].sum(axis=(1, 2)) >= 1)
        ############################## Predict TIS ##############################
        Y_pred_TIS[0].extend(Yps[0][is_expr, :, 2].flatten())  # TIS
        argsorted_y_pred_TIS = np.argsort(Y_pred_TIS[0][0:seq_len])[::-1]
        if TIS_score_cutoff < 1:  # using cutoff score
            for i in range(len(argsorted_y_pred_TIS)):
                if Y_pred_TIS[0][argsorted_y_pred_TIS[i]] < TIS_score_cutoff:
                    ind_threshold = i
                    break
        else: # threshold>=1: # indicating using top-threshold number
            ind_threshold = TIS_score_cutoff
        idx_pred = argsorted_y_pred_TIS[:int(ind_threshold)]
        pred_TIS_pos_score = [str(ind) + ',' + str(Y_pred_TIS[0][ind]) for ind in idx_pred]
        fhOut1.write(seqIn[idx * 2].strip('\n') + '\t' + '\t'.join(pred_TIS_pos_score) + '\n')
        # fhOut1.write(seqIn[idx * 2 + 1])
        ############################## Predict TTS ##############################
        Y_pred_TTS[0].extend(Yps[0][is_expr, :, 1].flatten())  # TTS
        argsorted_y_pred_TTS = np.argsort(Y_pred_TTS[0][0:seq_len])[::-1]
        if TTS_score_cutoff < 1:  # indicating using cutoff score
            for i in range(len(argsorted_y_pred_TTS)):
                if Y_pred_TTS[0][argsorted_y_pred_TTS[i]] < TTS_score_cutoff:
                    ind_threshold = i
                    break
        else:  # threshold>=1: # indicating using top-threshold number
            ind_threshold = TTS_score_cutoff
        idx_pred = argsorted_y_pred_TTS[:int(ind_threshold)]
        pred_TTS_pos_score = [str(ind) + ',' + str(Y_pred_TTS[0][ind]) for ind in idx_pred]
        fhOut2.write(seqIn[idx * 2].strip('\n') + '\t' + '\t'.join(pred_TTS_pos_score) + '\n')
        # fhOut2.write(seqIn[idx * 2 + 1])

    h5f.close()
    fhOut1.close()
    fhOut2.close()
    print("{:-^100}".format('TIS and TTS prediction done! Time used: %d seconds'%(time.time() - start_time)))



def main():

    start_time=time.time()

    args = get_options()
    if None in [args.I, args.t]:
        logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]]')
        exit()
    fnIn = args.I
    threshold_str = args.t

    TIS_score_cutoff = float(threshold_str.split(',')[0])
    TTS_score_cutoff = float(threshold_str.split(',')[1])
    if TIS_score_cutoff>=1:
        TIS_score_cutoff = int(TIS_score_cutoff)
    if TTS_score_cutoff >= 1:
        TTS_score_cutoff = int(TTS_score_cutoff)
    modelScale = '2000'
    modelsUsed = 'l1,l2,l3,l4,l5'
    show_dualORF = 0

    # Predict TISs and TTSs for each sequence
    fnIn1 = fnIn[0:-3] + '_predTIS_' + threshold_str.split(',')[0] + '.txt' # .fa file with predicted TISs
    fnIn2 = fnIn[0:-3] + '_predTTS_' + threshold_str.split(',')[1] + '.txt' # .fa file with predicted TTSs
    TIS_TTS_predictor(modelScale, modelsUsed, TIS_score_cutoff, TTS_score_cutoff, fnIn, 'TIS,TTS', 0, fnIn1, fnIn2)

    fnOut = fnIn[0:-3] + '_predORFs_' + '_'.join(threshold_str.split(',')) + '.txt' # .fa file with predicted ORFs
    fpr1=open(fnIn1, 'r')
    line=1
    # line_count=0
    predTIS_pos_score_list=[]
    # seq_list=[]
    header_list=[]
    while line:
        line=fpr1.readline()
        if line: # and line_count%2==0:
            header_list.append(line.split('\t')[0])
            words=line.strip().split('\t')
            predTIS_pos_score=[]
            if len(words)>1:
                TIS_pos_score_predict=[c.split(',') for c in words[1:]]
                TIS_pos_score_predict=[[int(c[0]),float(c[1])] for c in TIS_pos_score_predict]
                TIS_pos_score_predict=sorted(TIS_pos_score_predict,key=lambda x:x[1],reverse=True)
                if TIS_score_cutoff>=1:
                    predTIS_pos_score=TIS_pos_score_predict[0:int(TIS_score_cutoff)]
                else:
                    for c in TIS_pos_score_predict:
                        if c[1]>=TIS_score_cutoff:
                            predTIS_pos_score.append(c)
                        else:
                            break
            predTIS_pos_score_list.append(predTIS_pos_score)
            # seq_list.append(fpr1.readline().strip('\n'))
        # line_count+=2

    fpr2=open(fnIn2, 'r')
    line=1
    # line_count=0
    predTTS_pos_score_list=[]
    while line:
        line=fpr2.readline()
        if line: # and line_count%2==0:
            words=line.strip().split('\t')
            predTTS_pos_score=[]
            if len(words)>1:
                TTS_pos_score_predict=[c.split(',') for c in words[1:]]
                TTS_pos_score_predict=[[int(c[0]),float(c[1])] for c in TTS_pos_score_predict]
                TTS_pos_score_predict=sorted(TTS_pos_score_predict,key=lambda x:x[1],reverse=True)
                if TTS_score_cutoff>=1:
                    predTTS_pos_score=TTS_pos_score_predict[0:int(TTS_score_cutoff)]
                else:
                    for c in TTS_pos_score_predict:
                        if c[1]>=TTS_score_cutoff:
                            predTTS_pos_score.append(c)
                        else:
                            break
            predTTS_pos_score_list.append(predTTS_pos_score)
        # line_count+=1

    fhOut = open(fnOut, 'w')
    tot_predicted_ORF_num=0
    seq_num=len(predTIS_pos_score_list)
    for seq_i in range(seq_num):
        predicted_ORF_list=[]
        for TIS_j in range(len(predTIS_pos_score_list[seq_i])):
            [TIS_pos_j,TIS_score_j]=predTIS_pos_score_list[seq_i][TIS_j]
            for TTS_k in range(len(predTTS_pos_score_list[seq_i])):
                [TTS_pos_k,TTS_score_k]=predTTS_pos_score_list[seq_i][TTS_k]
                CDR_len=TTS_pos_k-TIS_pos_j
                if CDR_len>0 and not CDR_len%3:
                    tot_predicted_ORF_num+=1
                    predicted_ORF_list.append([[TIS_pos_j,TTS_pos_k],[TIS_score_j,TTS_score_k],TIS_score_j*TTS_score_k])
        predicted_ORF_list=sorted(predicted_ORF_list,key=lambda x:x[2],reverse=True)
        predicted_ORF_list=[c[0]+c[1] for c in predicted_ORF_list]
        predicted_ORF_list=[list(map(lambda x:str(x), c)) for c in predicted_ORF_list]
        predicted_ORF_str_list=[','.join(c) for c in predicted_ORF_list]
        content=header_list[seq_i]+'\t'+'\t'.join(predicted_ORF_str_list)
        if len(predicted_ORF_list)>0:
            fhOut.write(content + '\n')
            # fhOut.write(seq_list[seq_i] + '\n')
    fhOut.close()

    print("{:-^100}".format('Total sequence number: %d. Total predicted ORF number: %d' % (seq_num,tot_predicted_ORF_num)))
    print("{:-^100}".format('ALL done! Time used: %d seconds' % (time.time() - start_time)))

if __name__ == '__main__':
    main()
