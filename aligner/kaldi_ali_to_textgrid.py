# Script to take existing .ali files and generate textgrids from them.
# Used right now to deal with Kaldi-recipe-generated alignments (not MFA ones).

#from helper import make_path_safe, thirdparty_binary
#from ..multiprocessing import convert_ali_to_textgrids_kaldi
#import imp
#import sys
#sys.path.insert(0, "Users/mlml/Documents/GitHub/Montreal-Forced-Aligner/")
#from multiprocessing import convert_ali_to_textgrids_kaldi
#import multiprocessing as mp
import subprocess
import os
import shutil
import re
import sys
import imp
from decimal import Decimal

f, pathname, desc = imp.find_module('multiprocessing', sys.path[1:])
mp = imp.load_module('multiprocessing', f, pathname, desc)

from helper import make_path_safe, thirdparty_binary
#print(sys.path)

from textgrid import ctm_to_textgrid, parse_ctm

from config import *

from exceptions import CorpusError
from corpus import Corpus
from dictionary import Dictionary
#print(sys.path)

#import multiprocessing as mp


import subprocess
import os
import shutil
import re

def get_word_set():
    text_path = '/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/data/train_si84_hires/text'
    big_line = []
    with open(text_path, 'r') as fp:
        utts = fp.readlines()
        for line in utts:
            line = line.split(' ').pop(0)
            for word in line:
                if word not in big_line:
                    big_line.append(word)
    big_line = set(big_line)
    return big_line

def get_speaker_from_utt(utt):
    utt2spk_path = '/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/data/train_si84_hires/utt2spk'
    with open(utt2spk_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.split()
            if line[0] == utt:
                speaker = line[1]
                break
    return speaker

def parse_ctm_kaldi(ctm_path, dictionary, mode='word'):
    if mode == 'word':
        mapping = dictionary.reversed_word_mapping
    elif mode == 'phone':
        mapping = dictionary.reversed_phone_mapping
    file_dict = {}
    with open(ctm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line = line.split(' ')
            utt = line[0]
            begin = Decimal(line[2])
            duration = Decimal(line[3])
            end = begin + duration
            label = line[4]
            #speaker = corpus.utt_speak_mapping[utt]
            speaker = get_speaker_from_utt(utt)
            """if corpus.segments:
                filename = corpus.segments[utt]
                filename, utt_begin, utt_end = filename.split(' ')
                utt_begin = Decimal(utt_begin)
                if filename.endswith('_A') or filename.endswith('_B'):
                    filename = filename[:-2]
                begin += utt_begin
                end += utt_begin
            else:"""
            filename = utt

            try:
                label = mapping[int(label)]
            except KeyError:
                pass
            if mode == 'phone':
                for p in dictionary.positions:
                    if label.endswith(p):
                        label = label[:-1 * len(p)]
            if filename not in file_dict:
                file_dict[filename] = {}
            if speaker not in file_dict[filename]:
                file_dict[filename][speaker] = []
            file_dict[filename][speaker].append([begin, end, label])

    # Sort by begins
    for k, v in file_dict.items():
        for k2, v2 in v.items():
            file_dict[k][k2] = sorted(v2)

    return file_dict


def ali_to_textgrid_kaldi_func(ali_directory, model_directory, lang_dir, split_data_dir, sym2int_script, oov, job_name):  # pragma: no cover
    #text_int_path = os.path.join(corpus.split_directory, 'text.{}.int'.format(job_name))
    #log_path = os.path.join(model_directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    ali_path = os.path.join(ali_directory, 'ali.{}.gz'.format(job_name+1))
    model_path = os.path.join(model_directory, 'final.mdl')
    aligned_path = os.path.join(model_directory, 'aligned.{}'.format(job_name))
    word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(job_name))
    phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(job_name))
    phones_dir = os.path.join(lang_dir, 'phones')

    # Get integers
    log_path = os.path.join(ali_directory, 'log', 'sym2int.{}.log'.format(job_name))
    text_int_path = os.path.join(ali_directory, 'text.{}.int'.format(job_name))
    with open(log_path, 'w') as logf, open(text_int_path, 'w') as outf:
        sym2int_proc = subprocess.Popen([sym2int_script,
                                        '--map-oov', oov, '-f', '2-',
                                        os.path.join(lang_dir, 'words.txt'),
                                        os.path.join(split_data_dir, str(job_name+1), 'text')],
                                        stdout=outf, stderr=logf)
        sym2int_proc.communicate()

    frame_shift = 10/1000
    log_path = os.path.join(ali_directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    with open(log_path, 'w') as logf:
        lin_proc = subprocess.Popen(['/Users/mlml/Documents/Project/kaldi2/src/latbin/linear-to-nbest',
                                     "ark:gunzip -c " + ali_path + "|",
                                     "ark:" + text_int_path,
                                     '', '',
                                     'ark:-'],
                                     stdout=subprocess.PIPE, stderr=logf)
        align_proc = subprocess.Popen(['/Users/mlml/Documents/Project/kaldi2/src/latbin/lattice-align-words',
                                       os.path.join(phones_dir, 'word_boundary.int'), model_path,
                                       'ark:-', 'ark:' + aligned_path],
                                      stdin=lin_proc.stdout, stderr=logf)
        align_proc.communicate()

        subprocess.call(['/Users/mlml/Documents/Project/kaldi2/src/latbin/nbest-to-ctm',
                         '--frame-shift={}'.format(frame_shift),
                         'ark:' + aligned_path,
                         word_ctm_path],
                        stderr=logf)
        phone_proc = subprocess.Popen(['/Users/mlml/Documents/Project/kaldi2/src/latbin/lattice-to-phone-lattice', model_path,
                                       'ark:' + aligned_path, "ark:-"],
                                      stdout=subprocess.PIPE,
                                      stderr=logf)
        nbest_proc = subprocess.Popen(['/Users/mlml/Documents/Project/kaldi2/src/latbin/nbest-to-ctm',
                                       '--frame-shift={}'.format(frame_shift),
                                       "ark:-", phone_ctm_path],
                                      stdin=phone_proc.stdout,
                                      stderr=logf)
        nbest_proc.communicate()


def convert_ali_to_textgrids_kaldi(ali_directory, model_directory, lang_dir, split_data_dir, sym2int_script, oov, num_jobs):
    """
    Multiprocessing function that aligns based on the current model

    See:

    - http://kaldi-asr.org/doc/linear-to-nbest_8cc.html
    - http://kaldi-asr.org/doc/lattice-align-words_8cc.html
    - http://kaldi-asr.org/doc/lattice-to-phone-lattice_8cc.html
    - http://kaldi-asr.org/doc/nbest-to-ctm_8cc.html

    for more details
    on the Kaldi binaries this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/get_train_ctm.sh
    for the bash script that this function was based on.

    Parameters
    ----------
    output_directory : str
        Directory to write TextGrid files to
    model_directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object that has information about pronunciations
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object that has information about the dataset
    num_jobs : int
        The number of processes to use in calculation

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)

    """
    jobs = [(ali_directory, model_directory, lang_dir, split_data_dir, sym2int_script, oov, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs) as pool:
        r = False
        try:
            results = [pool.apply_async(ali_to_textgrid_kaldi_func, args=i) for i in jobs]
            output = [p.get() for p in results]
        except OSError as e:
            if hasattr(e, 'errno') and e.errorno == 24:
                r = True
            else:
                raise
    if r:
        raise (CorpusError(
            'There were too many files per speaker to process based on your OS settings.  Please try to split your data into more speakers.'))
    word_ctm = {}
    phone_ctm = {}
    for i in range(num_jobs):
        word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(i))
        phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(i))
        if not os.path.exists(word_ctm_path):
            continue
        parsed = parse_ctm_(word_ctm_path, corpus, dictionary, mode='word')
        for k, v in parsed.items():
            if k not in word_ctm:
                word_ctm[k] = v
            else:
                word_ctm[k].update(v)
        parsed = parse_ctm_(phone_ctm_path, corpus, dictionary, mode='phone')
        for k, v in parsed.items():
            if k not in phone_ctm:
                phone_ctm[k] = v
            else:
                phone_ctm[k].update(v)
    ctm_to_textgrid(word_ctm, phone_ctm, output_directory, corpus, dictionary)



#from ..multiprocessing import convert_ali_to_textgrids_kaldi



if __name__ == '__main__':
    kaldi_path = sys.argv[1]
    corpus_dir = '/Users/mlml/Documents/Project/train_si84'
    utils_dir = os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'utils')
    lang_dir = os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'data', 'lang')
    nnet_dir = os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'exp', 'nnet2_online', 'nnet_ms_a')
    data_dir = os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'data', 'train_si284_hires')
    ali_dir = os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'exp', 'nnet2_online', 'nnet_ms_a', 'ali_final')
    for (root, dirs, files) in os.walk(data_dir):
        for dir in dirs:
            if 'split' in dir:
                split_data_dir = os.path.join(data_dir, dir)
                num_jobs = int(dir.strip('split'))
                break
        break

    disambig_int_path = os.path.join(lang_dir, 'phones', 'disambig.int')
    tree_path = os.path.join(nnet_dir, 'tree')
    mdl_path = os.path.join(nnet_dir, 'final.mdl')
    L_fst_path = os.path.join(lang_dir, 'L.fst')

    dict_path = os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'data', 'local', 'dict_larger', 'lexicon.txt')

    sym2int_script = os.path.join(utils_dir, 'sym2int.pl')
    oov_path = os.path.join(lang_dir, 'oov.int')
    oov = ""
    with open(oov_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            oov = oov + line.strip()
    words_path = os.path.join(lang_dir, 'words.txt')

    scale_opts = '--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1'
    beam = 10
    retry_beam = 40

    corpus = Corpus(corpus_dir, os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'exp'),
                    speaker_characters=3,
                    num_jobs=30,
                    use_speaker_information=False,
                    ignore_exceptions=False)
    #dictionary = Dictionary(dict_path, os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'exp'), word_set=get_word_set())
    dictionary = Dictionary(dict_path, os.path.join(kaldi_path, 'egs', 'wsj', 's5', 'exp'), word_set=corpus.word_set)

    convert_ali_to_textgrids_kaldi(ali_dir, nnet_dir, lang_dir, split_data_dir, sym2int_script, oov, num_jobs)
