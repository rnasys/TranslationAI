## TranslationAI: An advanced deep learning framework for precise identification of translation initiation and termination sites within mature mRNA sequences

This package predicts translation initiation and termination sites with a given mature mRNA sequence. As an application, it can predict effect of genetic variants or RNA editing on translation product. 

### License
TranslationAI source code is provided under the [GPLv3 license](LICENSE). The trained models used by TranslationAI (located in this package at translationai/models) are provided under the [CC BY NC 4.0](LICENSE) license for academic and non-commercial use.

### Installation
TranslationAI can be installed from the [github repository](https://github.com/***/TranslationAI.git):

```sh
git clone https://github.com/rnasys/TranslationAI.git
cd TranslationAI
python setup.py install
```

TranslationAI requires ```tensorflow>=1.2.0```, which is best installed separately via pip or conda (see the [TensorFlow](https://www.tensorflow.org/) website for other installation options):

```sh
pip install tensorflow
# or
conda install tensorflow
```

### Usage
TranslationAI can be run from the command line (e.g.):

```sh
translationai -I translationai/examples/query_seq.fa -t 0.5,0.5
```

Required parameters:

 - ```-I```: Input .fa with query sequence(s).
 
 - ```-t```: Threshold k values for TIS/TTS prediction. If k>1, output the top-k TISs and TTSs in each sequence; if k<1, output the score>=k TISs and TTSs. e.g. 0.5,0.5


### Examples
A sample input file and the corresponding output file can be found at `examples/query_seq.fa` and `examples/query_seq_predTIS_0.1.txt` & `examples/query_seq_predTTS_0.1.txt` & `examples/query_seq_predORFs_0.1_0.1.txt` respectively. 

In the file for predicted TISs (e.g. `examples/query_seq_predTIS_0.1.txt`), each line represents a sequence and contains the following information: the sequence identifier, the predicted TIS position, the corresponding score. The TIS position and score are separated by a comma character. If multiple TISs were predicted for a sequence, they are ranked by the predicted scores and separated by a tab character. The format of each line is as follows:
`sequence_identifier	TIS_position_1,TIS_score_1   TIS_position_2,TIS_score_2   ...`

The file for predicted TTSs (e.g. `examples/query_seq_predTTS_0.1.txt`) is in the same format as the file for predicted TISs.

In the file for predicted ORFs (e.g. `examples/query_seq_predORFs_0.1_0.1.txt`), each line represents a sequence and contains the following information: the sequence identifier, the predicted TIS position, the predicted TTS position, the corresponding TIS score, the corresponding TTS score. The positions and scores are separated by a comma character. If multiple ORFs were predicted for a sequence, they are separated by a tab character. The format of each line is as follows:
`sequence_identifier	TIS_position_1,TTS_position_1,TIS_score_1,TTS_score_1   TIS_position_2,TTS_position_2,TIS_score_2,TTS_score_2   ...`

### Contact
Xiaojuan Fan: xiaojuanfan05@gmail.com
