import argparse
import os
from os.path import join, dirname
import PerceptualSimilarityModels as models
from util import util
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', type=str, default='./vg64/test_noise')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--all_pairs', action='store_true', help='turn on to test all N(N-1)/2 pairs, leave off to just do consecutive pairs (N-1)')
parser.add_argument('-N', type=int, default=None)
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu,version=opt.version)

# crawl directories
dirs = os.listdir(opt.dir)
dirs = [join(opt.dir, d) for d in dirs]

dists = []
for dir in dirs:
	files = os.listdir(dir)
	if(opt.N is not None):
		files = files[:opt.N]
	F = len(files)
	print(dir)
	print(files)

	for (ff,file) in enumerate(files[:-1]):
		img0 = util.im2tensor(util.load_image(os.path.join(dir,file))) # RGB image from [-1,1]
		if(opt.use_gpu):
			img0 = img0.cuda()

		if(opt.all_pairs):
			files1 = files[ff+1:]
		else:
			files1 = [files[ff+1],]

		for file1 in files1:
			print('(%s,%s)' % (file, file1))
			img1 = util.im2tensor(util.load_image(os.path.join(dir,file1)))

			if(opt.use_gpu):
				img1 = img1.cuda()

			# Compute distance
			dist01 = model.forward(img0,img1)
			print('(%s,%s): %.3f'%(file,file1,dist01))

			dists.append(dist01.item())

avg_dist = np.mean(np.array(dists))
stderr_dist = np.std(np.array(dists))/np.sqrt(len(dists))

print('Avg: %.5f +/- %.5f'%(avg_dist,stderr_dist))

output_dir = opt.dir
if output_dir[-1] == '/':
	output_dir = output_dir[:-1]
file_path = join(dirname(output_dir), "test_metrics.txt")
print(file_path)
with open(file_path, "a") as f:
	f.write("Diversity: %f + %f\n\n" % (avg_dist, stderr_dist))
