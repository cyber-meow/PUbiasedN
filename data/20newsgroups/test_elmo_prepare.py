import sys
import h5py
import allennlp.commands
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups


newsgroups_test = fetch_20newsgroups(subset='test')

n = len(newsgroups_test.data)

elmo = allennlp.commands.elmo.ElmoEmbedder(
    'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
    'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5', 0)


elmo2 = allennlp.commands.elmo.ElmoEmbedder(
    'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
    'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')


data = np.zeros([n, 9216])
f = h5py.File('20newsgroups_elmo_mmm_test.hdf5', 'w')
f.create_dataset('data', data=data)
f.close()


for i in range(n):
    A = word_tokenize(newsgroups_test.data[i])
    sys.stdout.write('processing document %d\r' % i)
    sys.stdout.flush()
    try:
        em = elmo.embed_batch([A])
    except:
        print('cuda fail')
        em = elmo2.embed_batch([A])
    em = np.concatenate(
            [np.mean(em[0], axis=1).flatten(),
             np.min(em[0], axis=1).flatten(),
             np.max(em[0], axis=1).flatten()])
    f = h5py.File('20newsgroups_elmo_mmm_test.hdf5', 'r+')
    f['data'][i] = em
    f.close()
