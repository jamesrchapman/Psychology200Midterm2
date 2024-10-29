import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def main():
	similarity = np.array([7,7,9,1,8,3,9,5,12,15,14,2,12,11,4])
	attraction = np.array([6,8,10,1,6,5,9,3,13,14,14,3,9,12,4])


	print("Similarity:\n",similarity)
	print("Attraction:\n",attraction)

	# Calculate Spearman's rho using my function
	print("my rho",spearman_rho(similarity,attraction))
	# Calculate Spearman's rho and p-value with scipy
	rho, p_value = spearmanr(similarity,attraction)


	print("Spearman's rho:", rho)
	print("P-value:", p_value)

	plt.figure(figsize=(8, 6))
	plt.scatter(similarity, attraction, color='purple', alpha=0.7, s=100, edgecolor='black')
	plt.xlabel('Similarity')
	plt.ylabel('Attraction')
	plt.title('Scatterplot of Similarity vs. Attraction')
	# Optionally, add a line of best fit if you're interested in the trend
	m, b = np.polyfit(similarity, attraction, 1)
	plt.plot(similarity, m*np.array(similarity) + b, color='darkorange', linestyle='--', linewidth=2, label='Trendline')

	plt.legend()
	plt.show()
	

def spearman_rho(X,Y):
	difference = X - Y
	difference_2 = np.square(difference)
	print("Difference:\n",difference)
	print("Difference Squared:\n",difference_2)
	N = len(X)
	rho=1-6*np.sum(difference_2)/(N**3-N)
	return rho

if __name__ == '__main__':
	main()


"""
Output:

Similarity:
 [ 7  7  9  1  8  3  9  5 12 15 14  2 12 11  4]
Attraction:
 [ 6  8 10  1  6  5  9  3 13 14 14  3  9 12  4]
Difference:
 [ 1 -1 -1  0  2 -2  0  2 -1  1  0 -1  3 -1  0]
Difference Squared:
 [1 1 1 0 4 4 0 4 1 1 0 1 9 1 0]
my rho 0.95
Spearman's rho: 0.9496859179096786
P-value: 6.334257697578373e-08

"""