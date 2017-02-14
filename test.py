import skimage.io as skio
import matplotlib.pyplot as plt
import pickle

with open('./data/training_image.pkl', 'rb') as read:
	a = pickle.load(read)
	print(a.shape)
#	img = skio.imread()
	skio.imshow(a)
	plt.show()
	
