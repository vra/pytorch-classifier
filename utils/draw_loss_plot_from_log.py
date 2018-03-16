from __future__ import print_function, division

import sys
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def parse_log(log_file):
	tr_loss = []
	te_loss = []
	tr_accu = []
	te_accu = []
	with open(log_file) as f:
		for line in f:
			if 'train Loss' in line:
				curr_tr_loss = float(line.strip().split('train Loss: ')[1].split(' ')[0])
				curr_tr_accu = float(line.strip().split('Acc: ')[-1])
				tr_loss.append(curr_tr_loss)
				tr_accu.append(curr_tr_accu)
			
			elif 'val Loss' in line:
				curr_te_loss = float(line.strip().split('val Loss: ')[1].split(' ')[0])
				curr_te_accu = float(line.strip().split('Acc: ')[-1])
				te_loss.append(curr_te_loss)
				te_accu.append(curr_te_accu)
	
	assert len(tr_loss) == len(tr_accu) and len(te_loss) == len(te_accu), 'length must equal'
	return np.array(tr_loss), np.array(te_loss), np.array(tr_accu), np.array(te_accu)


def draw_loss(tr_loss, te_loss, log_filename):
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(10, 4)
	ax = fig.gca()
	ax.set_yticks(np.arange(0, 1., 0.01))
	plt.grid(True)
	plt.subplot(1,2,1)
	c1, = plt.plot(tr_loss, label='training loss')
	c2, = plt.plot(te_loss, label='test loss')
	plt.xlabel('epochs')
	plt.ylabel('loss value')
	plt.legend(handles=[c1, c2])
	curr_file_dir = os.path.dirname(os.path.realpath(__file__))
#	plt.show()
	
def draw_accu(tr_accu, te_accu, log_filename):
	plt.subplot(1,2,2)
	c1, = plt.plot(tr_accu, label='training accuracy')
	c2, = plt.plot(te_accu, label='test accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend(handles=[c1, c2])
	curr_file_dir = os.path.dirname(os.path.realpath(__file__))
	fig_path = os.path.join(curr_file_dir, '../figures/'+log_filename.split('/')[-1].replace('log','png'))
	plt.savefig(fig_path)
	print('Figure of losses save to ',fig_path)
#	plt.show()



if __name__ == '__main__':
	print(os.path.dirname(os.path.realpath(__file__)))
	if len(sys.argv) is not 2:
		print('USAGE: python draw_plot_from_log.py /path/to/log/file')
		sys.exit(1)
	log_filename = sys.argv[1]
	tr_loss, te_loss, tr_accu, te_accu = parse_log(log_filename)
	draw_loss(tr_loss, te_loss, log_filename)
	draw_accu(tr_accu, te_accu, log_filename)

