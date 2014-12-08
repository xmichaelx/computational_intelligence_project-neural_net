import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from numpy import genfromtxt
import scipy.ndimage

filename = sys.argv[1]


def do_the_twist(filename, offset):
	data = genfromtxt(filename, delimiter=',')
	data[:,0] = np.remainder(data[:,0] + offset,1)
	data = data[np.argsort(data[:,0])]

	# removing most annoying outliers
	data[:,1] = scipy.ndimage.filters.median_filter(data[:,1], size = 7, mode='wrap') 
	scale = np.amax(data, axis=0)


	#plt.errorbar(data[:,0], data[:,1], data[:,2], fmt='.k', lw=1, ms=4, capsize=1.5)
	#plt.show()

	## now we have median filtered curve
	# and we're splitting to diffrent phase points every each 10

	sets = [ np.where(np.logical_and(data[:,0] > i/10.0,data[:,0] < (i+1)/10.0)) for i in range(10)]

	indexes = [np.amax(s) + 1 for s in sets[:-1]] 
	#print len(indexes)

	def section_error(data,indexes, i, size=1, plot = False):
		start,end = None, None
		if i == 0:
			start, end = 0, indexes[size-1]+1
		elif i == (len(indexes) - size +1):
			start, end = indexes[i-1], -1
		else:
			start, end = indexes[i-1], indexes[i-1+size]+1

		z = np.polyfit(data[start:end,0], data[start:end,1], 2)
		vals = np.poly1d(z)(data[start:end,0]) 
		if plot:
			plt.plot(data[start:end,0], vals, "b-")
	#		print z

		return sum(abs(vals - data[start:end,1])) 

	def sections_error(data, indexes):
		errors = [section_error(data,indexes, i, 1) for i in range(len(indexes) + 1)]
		return np.array(errors)


	def optimize(data, indexes):
		err0 = sections_error(data, indexes)
		for i in range(len(indexes)):
			index = indexes[i]
			indexes[i] += 1
			err2 = sections_error(data, indexes)
			while sum(err2 - err0) < 0:
				err0 = err2
				indexes[i] += 1
				err2 = sections_error(data, indexes)
				

				#print i, "room for improvement"
			indexes[i] -= 1

		for i in range(len(indexes)):
			index = indexes[i]
			indexes[i] -= 1
			err2 = sections_error(data, indexes)
			while sum(err2 - err0) < 0:
				err0 = err2
				indexes[i] -= 1
				err2 = sections_error(data, indexes)
				#print i, "room for improvement"
			indexes[i] += 1

		return indexes

	def plot_section(section):
		z = np.polyfit(section[:,0], section[:,1], 2)
		#print z
		vals = np.poly1d(z)(section[:,0]) 
		plt.plot(section[:,0], vals, "b-")

	def plot_sections(data, indexes):
		errors = [section_error(data,indexes, i, 1, True) for i in range(len(indexes) + 1)]

		return errors

	def prune_index(indexes, i, size):
		j = size - 1
		while (j > 0 ):
			if i < len(indexes):
				del indexes[i]
			j -= 1


	def pruning_pass(data, indexes,size):
		errors = [section_error(data,indexes, i, 1, False) for i in range(len(indexes) + 1)]
		summed_errors = []
		for i in range(len(indexes) + 1-(size-1)):
			summed_errors.append(sum(errors[i:i+size]))

		pruned_errors = np.array([section_error(data,indexes, i, size, False) for i in range(len(indexes) + 1-(size-1))])
		error_delta =  ((pruned_errors - summed_errors) / summed_errors)
		num = sum(error_delta < 0.2)

		t = list(np.argsort(error_delta)[:num])
		#print t
		for item in t:
			for overlap in range(item+1, item+size):
				if overlap in t:
					t.remove(overlap)

		t = sorted(t)
		t.reverse()

		removed_parabolas = 0
		for item in t:
			prune_index(indexes, item,size)
			removed_parabolas += size-1

		return removed_parabolas


	#print indexes
	#optimize(data, indexes)
	removed_parabolas = pruning_pass(data, indexes, 3)
	#optimize(data, indexes)

	plot_sections(data, indexes)


	plt.show()
	return 	sum(sections_error(data, indexes))
	#print removed_parabolas

	


print do_the_twist(filename, 0.05)

