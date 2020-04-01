## Author: Shashwat Banchhor 
## Contact: shashwatbanchhor12@gmail.com

## Elias
# sampling

## Separaet Building time and sampling time


import numpy as np
from numpy import linalg as LA
import random 
import heapq
from collections import defaultdict
import copy
import time
import torch
from prefix_codes import omega_coding, decode_omega_coding
from prefix_codes import delta_coding

random.seed(0)
###################### QUANTIZATION CODE ######################

# norm = torch.norm(flatten_grad) # 2-norm
# floor= torch.floor(flatten_grad.abs().div(norm)*qstates
#                 + torch.zeros(flatten_grad.shape, dtype=flatten_grad.dtype, device=flatten_grad.device).uniform_(0, 1))
# new_grad = torch.mul(flatten_grad.sign() * norm, floor/qstates)
# compress_grad = torch.where(torch.isinf(compress_grad), torch.zeros_like(compress_grad), new_grad)


# qstates = 255
# norm = LA.norm(flatten_grad)
# floor = np.floor(np.true_divide(np.absolute(flatten_grad),norm/qstates)+ np.random.uniform(0,1,flatten_grad.shape)) # Not taken care of data type
# new_grad = np.multiply(np.sign(flatten_grad)*norm, floor/qstates)
# whole_signed_new_grad = np.multiply(np.sign(flatten_grad), floor)
# new_grad = np.where(np.isinf(new_grad),np.zeros(new_grad.shape,dtype=float),new_grad)
###############################################################





def Huffman_Encode(frequency):
	heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
	heapq.heapify(heap)
	while len(heap) > 1:
		low = heapq.heappop(heap)
		high = heapq.heappop(heap)
		for value in low[1:]:
			value[1] = '0' + value[1]
		for value in high[1:]:
			value[1] = '1' +value[1]
		heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
	return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


def get_grad(filename):

	## Choosing a specific epoch
	flatten_grad = np.load(filename)
	qstates = 256
	norm = LA.norm(flatten_grad)
	floor = np.floor(np.true_divide(np.absolute(flatten_grad),norm/qstates)+ np.random.uniform(0,1,flatten_grad.shape)) # Not taken care of data type
	whole_signed_new_grad = np.multiply(np.sign(flatten_grad), floor)
	new_grad = np.multiply(np.sign(flatten_grad)*norm, floor/qstates)
	print("n/q",norm/qstates)
	new_grad = np.where(np.isinf(new_grad),np.zeros(new_grad.shape,dtype=float),new_grad)
	return whole_signed_new_grad , int(norm/qstates*1000) 


## Check for efficient implementation
def Run_Length_Encode(grads):
	out = []
	index = 0
	s = -10000
	for gradient_idx in range(len(grads)):
		if gradient_idx==0:
			s = int(grads[gradient_idx])
			count = 1

		else :
			if (s!= int(grads[gradient_idx])):
				# out.append(str(s))
				# index  += 1
				out.append(str(s)+'c'+str(count))
				index += 1
				s = int(grads[gradient_idx])
				count = 1

			else:
				count += 1
			
		if (gradient_idx==len(grads)-1):
			out.append(str(s)+'c'+str(count))

	out  = np.array(out)
	unique_elements, counts_elements = np.unique(out, return_counts=True)
	# print(unique_elements)
	# print(counts_elements)
	out = out.tolist()
	return out, len(out), unique_elements, counts_elements

def Run_Length_Encode_sparsity(grads):
	out = []
	index = 0
	s = -10000
	for gradient_idx in range(len(grads)):
		if gradient_idx==0:
			s = int(grads[gradient_idx])
			count = 1

		else :
			if (s!= int(grads[gradient_idx])):
				# out.append(str(s))
				# index  += 1
				if(s==0):    
					out.append(str(s)+'c'+str(count))
					index += 1
					s = int(grads[gradient_idx])
					count = 1
				else:
					out.append(str(s))
					index += 1
					s = int(grads[gradient_idx])
					count = 1 

			else:
				if(s==0):
					count += 1
				else:
					out.append(str(s))
					index += 1
					s = int(grads[gradient_idx])
					count = 1 
			
		if (gradient_idx==len(grads)-1):
			if(s==0):
				out.append(str(s)+'c'+str(count))
			else:
				out.append(str(s))
	out  = np.array(out)
	unique_elements, counts_elements = np.unique(out, return_counts=True)
	# print(unique_elements)
	# print(counts_elements)
	out = out.tolist()
	return out, len(out), unique_elements, counts_elements

def Run_Length_Encode_efficient(grads):
	out = []
	index = 0
	s = -10000
	frequency = {}
	for gradient_idx in range(len(grads)):
		if gradient_idx==0:
			s = int(grads[gradient_idx])
			count = 1

		else :
			if (s!= int(grads[gradient_idx])):
				# out.append(str(s))
				# index  += 1
				out.append(str(s)+'c'+str(count))
				try:
					frequency[str(s)+'c'+str(count)] += 1
				except KeyError as e:
					frequency[str(s)+'c'+str(count)] = 1
				index += 1
				s = int(grads[gradient_idx])
				count = 1

			else:
				count += 1
			
		if (gradient_idx==len(grads)-1):
			out.append(str(s)+'c'+str(count))
			index+=1
			try:
				frequency[str(s)+'c'+str(count)] += 1
			except KeyError as e:
				frequency[str(s)+'c'+str(count)] = 1

	# out  = np.array(out)
	# unique_elements, counts_elements = np.unique(out, return_counts=True)
	# print(unique_elements)
	# print(counts_elements)
	# out = out.tolist()
	return out, index, frequency

def Run_Length_Encode_efficient_16bit(grads):
	start_time = time.time()
	encodings = {}
	for i in range(256):
		encodings[i]  = "1111111111111111"
	encoded_doc = []
	out = []
	index = 0
	s = -10000
	frequency = {}
	for gradient_idx in range(len(grads)):
		if gradient_idx==0:
			s = int(grads[gradient_idx])
			count = 1

		else :
			if (s!= int(grads[gradient_idx])):
				# out.append(str(s))
				# index  += 1
				out.append(str(s)+'c'+str(count))
				encoded_doc.append(encodings[1])
				encoded_doc.append(encodings[2])
				try:
					frequency[str(s)+'c'+str(count)] += 1
				except KeyError as e:
					frequency[str(s)+'c'+str(count)] = 1
				index += 1
				s = int(grads[gradient_idx])
				count = 1

			else:
				count += 1
			
		if (gradient_idx==len(grads)-1):
			out.append(str(s)+'c'+str(count))
			encoded_doc.append(encodings[1])
			encoded_doc.append(encodings[2])
			index+=1
			try:
				frequency[str(s)+'c'+str(count)] += 1
			except KeyError as e:
				frequency[str(s)+'c'+str(count)] = 1

	# out  = np.array(out)
	# unique_elements, counts_elements = np.unique(out, return_counts=True)
	# print(unique_elements)
	# print(counts_elements)
	# out = out.tolist()
	end_time = time.time()
	encoded_doc = "".join(encoded_doc)
	
	print ("16-bit encoding L :", len(encoded_doc), "Time: ", end_time - start_time)
	return out, index, frequency, encoded_doc 


def __recursive_decode(s, n):
	if s[0]=="0":
		return [n, s[1:]]
	else:
		m = int(s[:n+1], 2)
		return __recursive_decode(s[n+1:], m)

def decode(s):
   

	return __recursive_decode(s, 1)

def Run_Length_Encode_efficient_16bit_omega(grads):
	start_time = time.time()
	# encodings = {}
	# for i in range(256):
	#     encodings[i]  = "1111111111111111"
	encoded_doc = []
	out = []
	index = 0
	s = -10000
	frequency = {}
	for gradient_idx in range(len(grads)):
		if gradient_idx==0:
			s = int(grads[gradient_idx])
			count = 1

		else :
			if (s!= int(grads[gradient_idx])):
				# out.append(str(s))
				# index  += 1
				out.append(str(s)+'c'+str(count))
				if(s<0):
					encoded_doc.append(omega_coding(2*(-s)))
				else:
					encoded_doc.append(omega_coding(2*s+1))
				encoded_doc.append(omega_coding(count))
				try:
					frequency[str(s)+'c'+str(count)] += 1
				except KeyError as e:
					frequency[str(s)+'c'+str(count)] = 1
				index += 1
				s = int(grads[gradient_idx])
				count = 1

			else:
				count += 1
			
		if (gradient_idx==len(grads)-1):
			out.append(str(s)+'c'+str(count))
			# encoded_doc.append(omega_coding(s))
			if(s<0):
					encoded_doc.append(omega_coding(2*(-s)))
			else:
					encoded_doc.append(omega_coding(2*s+1))
			encoded_doc.append(omega_coding(count))
			index+=1
			try:
				frequency[str(s)+'c'+str(count)] += 1
			except KeyError as e:
				frequency[str(s)+'c'+str(count)] = 1

	# out  = np.array(out)
	# unique_elements, counts_elements = np.unique(out, return_counts=True)
	# print(unique_elements)
	# print(counts_elements)
	# out = out.tolist()
	end_time = time.time()
	encoded_doc_str = "".join(encoded_doc)
	encoded_doc = torch.zeros([len(encoded_doc_str),],  dtype=torch.int8)
	doc_id = 0
	for bit in encoded_doc_str:
		encoded_doc[doc_id] = int(bit)
		doc_id += 1
	# print ("Elias Omega encoding L :", len(encoded_doc), "Time: ", end_time - start_time)
	return out, index, frequency, encoded_doc 

def Run_Length_Encode_efficient_16bit_delta(grads):
	start_time = time.time()
	# encodings = {}
	# for i in range(256):
	#     encodings[i]  = "1111111111111111"
	encoded_doc = []
	out = []
	index = 0
	s = -10000
	frequency = {}
	for gradient_idx in range(len(grads)):
		if gradient_idx==0:
			s = int(grads[gradient_idx])
			count = 1

		else :
			if (s!= int(grads[gradient_idx])):
				# out.append(str(s))
				# index  += 1
				out.append(str(s)+'c'+str(count))
				encoded_doc.append(delta_coding(s))
				encoded_doc.append(delta_coding(count))
				try:
					frequency[str(s)+'c'+str(count)] += 1
				except KeyError as e:
					frequency[str(s)+'c'+str(count)] = 1
				index += 1
				s = int(grads[gradient_idx])
				count = 1

			else:
				count += 1
			
		if (gradient_idx==len(grads)-1):
			out.append(str(s)+'c'+str(count))
			encoded_doc.append(delta_coding(s))
			encoded_doc.append(delta_coding(count))
			index+=1
			try:
				frequency[str(s)+'c'+str(count)] += 1
			except KeyError as e:
				frequency[str(s)+'c'+str(count)] = 1

	# out  = np.array(out)
	# unique_elements, counts_elements = np.unique(out, return_counts=True)
	# print(unique_elements)
	# print(counts_elements)
	# out = out.tolist()
	end_time = time.time()
	encoded_doc = "".join(encoded_doc)
	
	print ("Elias Delta encoding L :", len(encoded_doc), "Time: ", end_time - start_time)
	return out, index, frequency, encoded_doc 

def Run_Length_Encode_efficient_update(grads, encoding,k):
	out = []
	encoded_doc = []
	index = 0
	s = -10000
	frequency = {}
	for gradient_idx in range(len(grads)):
		if gradient_idx==0:
			s = int(grads[gradient_idx])
			count = 1

		else :
			if (s!= int(grads[gradient_idx])):
				# out.append(str(s))
				# index  += 1
				# out.append(str(s)+'c'+str(count))
				if(s==0 and count <= k):
					encoded_doc.append(encoding[str(s)+'c'+str(count)])
				else:
					if (s==0):
						encoded_doc.append(encoding[str(s)+'c'+str(1)]*count)
					else:
						encoded_doc.append(encoding[str(s)]*count)
				# try:
				#     frequency[str(s)+'c'+str(count)] += 1
				# except KeyError as e:
				#     frequency[str(s)+'c'+str(count)] = 1
				index += 1
				s = int(grads[gradient_idx])
				count = 1

			else:
				count += 1
			
		if (gradient_idx==len(grads)-1):
			# out.append(str(s)+'c'+str(count))
			if(s==0):
				encoded_doc.append(encoding[str(s)+'c'+str(count)])
			else:
				encoded_doc.append(encoding[str(s)]*count)
			index+=1
			# try:
			#     frequency[str(s)+'c'+str(count)] += 1
			# except KeyError as e:
			#     frequency[str(s)+'c'+str(count)] = 1

	# out  = np.array(out)
	# unique_elements, counts_elements = np.unique(out, return_counts=True)
	# print(unique_elements)
	# print(counts_elements)
	# out = out.tolist()
	return "".join(encoded_doc)


## Omega


##################### RUN LENGTH HUFFMAN #####################
def run_length_huffman(quantized_grads):
	start_time = time.time()
	# document, doc_len, frequency =  Run_Length_Encode_efficient(quantized_grads)

	document, doc_len, frequency =  Run_Length_Encode_efficient(quantized_grads)
	sample_time_e = time.time()

	total_freq = 0
	for key in frequency:
		total_freq += frequency[key]
	# for key in frequency:
		# print("K:", key, "Freq", frequency[key]/total_freq)
	# frequency = {}
	# for x in range(len(unique_elements)):
	#     frequency[unique_elements[x]] = counts_elements[x]
	# print (type(unique_elements) , len(counts_elements))
	H_time_s = time.time()
	run_length_huff = Huffman_Encode(frequency)
	encodings  = {}
	for i in run_length_huff:
		encodings[i[0]] = i[1]
	H_time_e = time.time()


	encode_time_s = time.time()
	encoded_doc = []
	for i in range(doc_len):
		encoded_doc.append(encodings[document[i]])
	encoded_doc  = "".join(encoded_doc)
	encode_time_e = time.time()

	end_time = time.time()


	run_code_length = 0
	for i in run_length_huff:
		# print(i[0].ljust(10) + str(frequency[i[0]]).ljust(10) + i[1])
		run_code_length += frequency[i[0]]*len(i[1])
	# print ("Run Length Huffman L* :",run_code_length,  "Time: ", end_time - start_time)
	print ("Run Length Huffman L* :",run_code_length,  "SampleTime: ", sample_time_e - start_time, "Build Huff Time: ", H_time_e- H_time_s, "Encode Time: ", encode_time_e - encode_time_s)
   
	return run_code_length



def run_length_huffman_core(quantized_grads, prev_count=0, magnitude = 0):
	start_time = time.time()
	# document, doc_len, frequency =  Run_Length_Encode_efficient(quantized_grads)

	document, doc_len, frequency =  Run_Length_Encode_efficient(quantized_grads)
	sample_time_e = time.time()

	total_freq = 0
	for key in frequency:
		total_freq += frequency[key]
	# for key in frequency:
		# print("K:", key, "Freq", frequency[key]/total_freq)
	# frequency = {}
	# for x in range(len(unique_elements)):
	#     frequency[unique_elements[x]] = counts_elements[x]
	# print (type(unique_elements) , len(counts_elements))
	H_time_s = time.time()
	run_length_huff = Huffman_Encode(frequency)
	encodings  = {}
	swap_encodings = {}
	for i in run_length_huff:
		encodings[i[0]] = i[1]
		swap_encodings[i[1]] = i[0]
	H_time_e = time.time()


	encode_time_s = time.time()
	# encoded_doc = []
	encoded_doc = torch.zeros([15000000,],  dtype=torch.int8)
	count = prev_count
	# encoded_doc[count] = 123
	# count += 1
	for i in range(doc_len):
		# encoded_doc.append(encodings[document[i]])
		encd = encodings[document[i]]
		# print("ENCD:: ",encd)
		for char in encd:
			encoded_doc[count] = ord(char)
			count += 1
		encoded_doc[count] = 61
		count += 1
	# print("DOC_LEN", doc_len, count)
	# encoded_doc  = "".join(encoded_doc)
	encode_time_e = time.time()

	end_time = time.time()


	run_code_length = 0
	for i in run_length_huff:
		# print(i[0].ljust(10) + str(frequency[i[0]]).ljust(10) + i[1])
		run_code_length += frequency[i[0]]*len(i[1])
	# print ("Run Length Huffman L* :",run_code_length,  "Time: ", end_time - start_time)
	# print ("Run Length Huffman L* :",run_code_length,  "SampleTime: ", sample_time_e - start_time, "Build Huff Time: ", H_time_e- H_time_s, "Encode Time: ", encode_time_e - encode_time_s)
   
	return document, run_code_length, encodings, swap_encodings, encoded_doc[:count]
##############################################################



##################### SAMPLE RUN LENGTH HUFFMAN #####################
def sample_run_length_huffman(quantized_grads):
	sample_time_s = time.time()
	document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	sample_frequency = sample_frequencies(unique_elements, document)
	# print(sample_frequency)
	frequency = {}
	for x in range(len(unique_elements)):
		frequency[unique_elements[x]] = counts_elements[x]
	sample_time_e = time.time()
	# print(frequency)
	# print(unique_elements)
	# print (type(unique_elements) , len(counts_elements))
	H_time_s = time.time()
	run_length_huff = Huffman_Encode(sample_frequency)
	H_time_e = time.time()

	encode_time_s = time.time()
	run_code_length = 0
	for i in run_length_huff:
		# print(i[0].ljust(10) + str(frequency[i[0]]).ljust(10) + i[1])
		run_code_length += frequency[i[0]]*len(i[1])
	encode_time_e = time.time()
	print ("Sample Run Length Huffman L* :",run_code_length, "Sample Time: ", sample_time_e- sample_time_s, "Build Huff Tree: ", H_time_e - H_time_s, "Encode Time: ", encode_time_e - encode_time_s)
	return run_code_length
##############################################################






##################### SAMPLE HUFFMAN #####################
def sample_frequencies(unique_qstates, quantized_grads):   
	S = 10000
	# Preemptive Smoothing #######################
	sample_frequency = {}
	for key in unique_qstates :
		sample_frequency[str(int(key))] = 1
			#############################################
	for x in range(S):
		### Sampling done uniformly at random ###
		idx = random.randint(0,len(quantized_grads)-1)
		# if (quantized_grads[idx] not in sample_frequency):
		#     sample_frequency[str(quantized_grads[idx])] = 1
		# else:
		sample_frequency[str(int(quantized_grads[idx]))] +=  1
	## probability of getting 0 in the document
	# print(type(frequency), type(sample_frequency))
	# print(sample_frequency)
	# exit(0)
	return sample_frequency


def base_frequencies(unique_qstates, counts_qstates, quantized_grads):
	base_frequency = {}
	for x in range(len(unique_qstates)):
		base_frequency[str(unique_qstates[x])] = counts_qstates[x]
	return base_frequency   

## NO SPARSITY ############
def sample_huffman_no_sparsity(quantized_grads):
	# document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	unique_qstates =  torch.unique(quantized_grads)
	#base_frequency = base_frequencies(unique_qstates, counts_qstates, quantized_grads)
	sample_frequency = sample_frequencies(unique_qstates, quantized_grads)
	sample_no_sparse_huff = Huffman_Encode(sample_frequency)

	encodings  = {}
	swap_encodings = {}
	for i in sample_no_sparse_huff:
		encodings[i[0]] = i[1]
		swap_encodings[i[1]] = i[0]
	
	encoded_doc = encode( quantized_grads, encodings)
	# print("character".ljust(10) + "Weight".ljust(10) + "Huffman Code")
	# sample_no_sparse_code_length = 0
	# for i in sample_no_sparse_huff:
	#     # print(i[0].ljust(10) + str(sample_frequency[i[0]]).ljust(10) + i[1])
	#     sample_no_sparse_code_length += base_frequency[i[0]]*len(i[1])
	# print ("Sample_no_sparse_Huffman L* :",sample_no_sparse_code_length)
	# return sample_no_sparse_code_length

	return swap_encodings, encoded_doc
###########################


## SPARSITY ###############
def sample_huffman_with_sparsity(quantized_grads):
	k = 200
	document, doc_len, unique_elements, counts_elements =  Run_Length_Encode_sparsity(quantized_grads)
	unique_qstates = torch.unique(quantized_grads)
	print(unique_qstates)
	# for i in range(len(unique_qstates)):
	#     print(unique_qstates[i],":",counts_qstates[i], end="    ")
	# unique_elements, counts_elements = np.unique(document, return_counts=True)
	# print("::::::::::::::::::")
	# for i in range(len(unique_elements)):
	#     print(unique_elements[i],":",counts_elements[i], end="    ")
	sample_frequency = sample_frequencies(unique_qstates, quantized_grads)
	# print(sample_frequency)
	# exit(0)
	total_freq = 0
	for key in sample_frequency.keys():
		total_freq += sample_frequency[key]

	for key in sample_frequency.keys():
		sample_frequency[key] = sample_frequency[key]/total_freq


	### Probability for n-zeros
	for repetitions in range(1,k):   
		sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*(1- sample_frequency[str(0)])
	sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)#*(1- sample_frequency[str(0)])
	del sample_frequency["0"]



	sample_huff = Huffman_Encode(sample_frequency)

	sample_sparse_code_length = 0
	sample_encoding = {}
	encodings  = {}
	swap_encodings = {}
	for i in sample_huff:
		# print(i[0].ljust(10) + str(sample_frequency[i[0]]).ljust(30) + i[1])
		sample_encoding[i[0]] = (sample_frequency[i[0]], len(i[1]))
		# sample_sparse_code_length += base_frequency[i[0]]*len(i[1])
		encodings[i[0]] = i[1]
		swap_encodings[i[1]] = i[0]

	# print(document)
	# exit(0)

	# for run in document:
	#     sparse = run.split("c")
	#     sparse[0] = int(sparse[0])
	#     # sparse[1] = int(sparse[1])
	#     if (sparse[0]==0 and int(sparse[1])<=k):
			
	#         sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(sparse[1])][1]
	#     else:
	#         ####  We can improve more on this  ######
	#         if (sparse[0]==0):
	#             sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(1)][1] * sparse[1]
	#         else:
	#             sample_sparse_code_length += sample_encoding[str(sparse[0])][1] * 1
		


	# print ("Sample Sparse Huffman L :",sample_sparse_code_length) #, "SH/RLH",sample_sparse_code_length/run_code_length)
	# return sample_sparse_code_length
	

	encoded_doc = encode( document, encodings)

	return swap_encodings, encoded_doc
###########################


def sample_huffman_with_sparsity_efficient(quantized_grads):
	start_time = time.time()
	k = 200
	# document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	# unique_qstates,counts_qstates = np.unique(quantized_grads, return_counts=True)
	# sample_frequency = sample_frequencies(unique_qstates, quantized_grads)
	S = 10000
	# Preemptive Smoothing #######################
	total_freq = 0
	sample_frequency = {}
	for key in range(-127, 129,1) :
		sample_frequency[str(key)] = 1
		total_freq += 1
			#############################################
	
	for x in range(S):
		### Sampling done uniformly at random ###
		idx = random.randint(0,len(quantized_grads)-1)
		# if (quantized_grads[idx] not in sample_frequency):
		#     sample_frequency[str(quantized_grads[idx])] = 1
		# else:
		sample_frequency[str(quantized_grads[idx])] +=  1
		total_freq += 1

	
	# for key in sample_frequency.keys():
	#     total_freq += sample_frequency[key]

	for key in sample_frequency.keys():
		sample_frequency[key] = sample_frequency[key]/total_freq
		# print("Initial Freq", "char", key, "prob", sample_frequency[key])
	# ### Probability for n-zeros
	# for repetitions in range(1,k):   
	#     sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*(1- sample_frequency[str(0)])
	# sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)#*(1- sample_frequency[str(0)])
	# del sample_frequency["0"]

	p_zero = sample_frequency[str(0)]
	for repetitions in range(1,k):   
		sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*((1- sample_frequency[str(0)]))/(1 + sample_frequency[str(0)])
	sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)/(1 + sample_frequency[str(0)])#*(1- sample_frequency[str(0)])
	del sample_frequency["0"]
	## Normalising
	prob = 0
	for key in sample_frequency.keys():
		if (key[0]!='0'):
			sample_frequency[key] /= (1- p_zero**2)
		# print("Final Freq", "char", key, "prob", sample_frequency[key])
		prob += sample_frequency[key]

	print("Prob",prob)

	sample_huff = Huffman_Encode(sample_frequency)
	encodings  = {}
	for i in sample_huff:
		encodings[i[0]] = i[1]

	encoded_doc = Run_Length_Encode_efficient_update(quantized_grads, encodings, k)
	end_time = time.time()

	sample_sparse_code_length = 0
	sample_encoding = {}
	for i in sample_huff:
		# print(i[0].ljust(10) + str(sample_frequency[i[0]]).ljust(30) + i[1])
		sample_encoding[i[0]] = (sample_frequency[i[0]], len(i[1]))
		# sample_sparse_code_length += base_frequency[i[0]]*len(i[1])

	document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	for run in document:
		sparse = run.split("c")
		sparse[0] = int(sparse[0])
		sparse[1] = int(sparse[1])
		if (sparse[0]==0 and sparse[1]<=k):
			
			sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(sparse[1])][1]
		else:
			####  We can improve more on this  ######
			if (sparse[0]==0):
				sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(1)][1] * sparse[1]
			else:
				sample_sparse_code_length += sample_encoding[str(sparse[0])][1] * sparse[1]
		


	# print ("Efficient Sample Sparse Huffman L :",sample_sparse_code_length, len(encoded_doc), "Time: ", end_time - start_time) #, "SH/RLH",sample_sparse_code_length/run_code_length)
	return sample_sparse_code_length

def sample_huffman_with_sparsity_efficient_old(quantized_grads):
	start_time = time.time()
	k = 200
	# document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	# unique_qstates,counts_qstates = np.unique(quantized_grads, return_counts=True)
	# sample_frequency = sample_frequencies(unique_qstates, quantized_grads)
	S = 10000
	# Preemptive Smoothing #######################
	total_freq = 0
	sample_frequency = {}
	for key in range(-127, 129,1) :
		sample_frequency[str(key)] = 1
		total_freq += 1
			#############################################
	
	for x in range(S):
		### Sampling done uniformly at random ###
		idx = random.randint(0,len(quantized_grads)-1)
		# if (quantized_grads[idx] not in sample_frequency):
		#     sample_frequency[str(quantized_grads[idx])] = 1
		# else:
		sample_frequency[str(quantized_grads[idx])] +=  1
		total_freq += 1

	# for key in sample_frequency.keys():
	#     total_freq += sample_frequency[key]

	for key in sample_frequency.keys():
		sample_frequency[key] = sample_frequency[key]/total_freq


	# ### Probability for n-zeros
	for repetitions in range(1,k):   
		sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*(1- sample_frequency[str(0)])
	sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)#*(1- sample_frequency[str(0)])
	del sample_frequency["0"]
	sample_time_e = time.time()
	# p_zero = sample_frequency[str(0)]
	# for repetitions in range(1,k):   
	#     sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*((1- sample_frequency[str(0)]))/(1 + sample_frequency[str(0)])
	# sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)/(1 + sample_frequency[str(0)])#*(1- sample_frequency[str(0)])
	# del sample_frequency["0"]
	## Normalising
	# prob = 0
	# for key in sample_frequency.keys():
	#     if (key[0]!='0'):
	#         sample_frequency[key] /= (1- p_zero**2)
	#     prob += sample_frequency[key]

	# print("Prob",prob)


	H_time_s = time.time()
	sample_huff = Huffman_Encode(sample_frequency)
	encodings  = {}
	for i in sample_huff:
		encodings[i[0]] = i[1]
	H_time_e = time.time()


	encode_time_s = time.time()
	encoded_doc = Run_Length_Encode_efficient_update(quantized_grads, encodings, k)
	encode_time_e = time.time()

	end_time = time.time()




	sample_sparse_code_length = 0
	sample_encoding = {}
	for i in sample_huff:
		# print(i[0].ljust(10) + str(sample_frequency[i[0]]).ljust(30) + i[1])
		sample_encoding[i[0]] = (sample_frequency[i[0]], len(i[1]))
		# sample_sparse_code_length += base_frequency[i[0]]*len(i[1])

	document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	for run in document:
		sparse = run.split("c")
		sparse[0] = int(sparse[0])
		sparse[1] = int(sparse[1])
		if (sparse[0]==0 and sparse[1]<=k):
			
			sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(sparse[1])][1]
		else:
			####  We can improve more on this  ######
			if (sparse[0]==0):
				sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(1)][1] * sparse[1]
			else:
				sample_sparse_code_length += sample_encoding[str(sparse[0])][1] * sparse[1]
		


	# print ("Efficient Sample Sparse Huffman L :",sample_sparse_code_length, len(encoded_doc), "Time: ", end_time - start_time) #, "SH/RLH",sample_sparse_code_length/run_code_length)
	print ("Efficient Sample Sparse Huffman L :",sample_sparse_code_length, len(encoded_doc), "SampleTime: ", sample_time_e - start_time, "Build Huff Tree: ", H_time_e - H_time_s, "Encode Time: ", encode_time_e - encode_time_s) #, "SH/RLH",sample_sparse_code_length/run_code_length)

	return sample_sparse_code_length


def sample_huffman_with_sparsity_efficient_sign_check(quantized_grads, prev_quantized_grads, initial):
	start_time = time.time()
	if (initial!=1):
		sign_change = 0
		for i in range(len(quantized_grads)):
			# if (quantized_grads[i] != prev_quantized_grads[i]):
			#     diff += 1 
			# print(quantized_grads[i], prev_quantized_grads[i])
			if ((quantized_grads[i] > 0 and  prev_quantized_grads[i] <0)  or (quantized_grads[i] < 0 and  prev_quantized_grads[i] > 0) ):
				sign_change += 1 

		# print(quantized_grads[:100])
		# print(prev_quantized_grads[:100])
		
			# Diff
		# print("Iter ",str(_iter_-1),"-",str(_iter_), " Diff count: ", diff, "Total Len: ", len(quantized_grads))    

		percent = sign_change/len(quantized_grads)
			# Sign Change
		print("Iter ",str(_iter_-1),"-",str(_iter_), " Sign Change count: ", sign_change, "Total Len: ", len(quantized_grads), "percent", sign_change/len(quantized_grads))    
		if (percent < 0.1): 
			end_time = time.time()
			print(end_time - start_time)
	prev_quantized_grads = np.copy(quantized_grads)
	k = 200
	# document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	# unique_qstates,counts_qstates = np.unique(quantized_grads, return_counts=True)
	# sample_frequency = sample_frequencies(unique_qstates, quantized_grads)
	S = 10000
	# Preemptive Smoothing #######################
	total_freq = 0
	sample_frequency = {}
	for key in range(-127, 129,1) :
		sample_frequency[str(key)] = 1
		total_freq += 1
			#############################################
	
	for x in range(S):
		### Sampling done uniformly at random ###
		idx = random.randint(0,len(quantized_grads)-1)
		# if (quantized_grads[idx] not in sample_frequency):
		#     sample_frequency[str(quantized_grads[idx])] = 1
		# else:
		sample_frequency[str(quantized_grads[idx])] +=  1
		total_freq += 1

	
	# for key in sample_frequency.keys():
	#     total_freq += sample_frequency[key]

	for key in sample_frequency.keys():
		sample_frequency[key] = sample_frequency[key]/total_freq
		# print("Initial Freq", "char", key, "prob", sample_frequency[key])
	# ### Probability for n-zeros
	# for repetitions in range(1,k):   
	#     sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*(1- sample_frequency[str(0)])
	# sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)#*(1- sample_frequency[str(0)])
	# del sample_frequency["0"]

	p_zero = sample_frequency[str(0)]
	for repetitions in range(1,k):   
		sample_frequency["0c"+str(repetitions)] = ((sample_frequency[str(0)])**repetitions)*((1- sample_frequency[str(0)]))/(1 + sample_frequency[str(0)])
	sample_frequency["0c"+str(k)] = ((sample_frequency[str(0)])**k)/(1 + sample_frequency[str(0)])#*(1- sample_frequency[str(0)])
	del sample_frequency["0"]
	## Normalising
	prob = 0
	for key in sample_frequency.keys():
		if (key[0]!='0'):
			sample_frequency[key] /= (1- p_zero**2)
		# print("Final Freq", "char", key, "prob", sample_frequency[key])
		prob += sample_frequency[key]

	print("Prob",prob)

	sample_huff = Huffman_Encode(sample_frequency)
	encodings  = {}
	for i in sample_huff:
		encodings[i[0]] = i[1]

	encoded_doc = Run_Length_Encode_efficient_update(quantized_grads, encodings, k)
	end_time = time.time()

	sample_sparse_code_length = 0
	sample_encoding = {}
	for i in sample_huff:
		# print(i[0].ljust(10) + str(sample_frequency[i[0]]).ljust(30) + i[1])
		sample_encoding[i[0]] = (sample_frequency[i[0]], len(i[1]))
		# sample_sparse_code_length += base_frequency[i[0]]*len(i[1])

	document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	for run in document:
		sparse = run.split("c")
		sparse[0] = int(sparse[0])
		sparse[1] = int(sparse[1])
		if (sparse[0]==0 and sparse[1]<=k):
			
			sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(sparse[1])][1]
		else:
			####  We can improve more on this  ######
			if (sparse[0]==0):
				sample_sparse_code_length += sample_encoding[str(sparse[0])+"c"+str(1)][1] * sparse[1]
			else:
				sample_sparse_code_length += sample_encoding[str(sparse[0])][1] * sparse[1]
		


	print ("Efficient Sample Sparse Huffman L :",sample_sparse_code_length, len(encoded_doc), "Time: ", end_time - start_time) #, "SH/RLH",sample_sparse_code_length/run_code_length)
	return sample_sparse_code_length

##########################################################



##################### BASE HUFFMAN #######################
def base_huffman(quantized_grads):
	# document, doc_len, unique_elements, counts_elements =  Run_Length_Encode(quantized_grads)
	unique_qstates,counts_qstates = np.unique(quantized_grads, return_counts=True)
	base_frequency = base_frequencies(unique_qstates, counts_qstates, quantized_grads)
	base_huff = Huffman_Encode(base_frequency)
	# print("character".ljust(10) + "Weight".ljust(10) + "Huffman Code")
	base_code_length = 0
	for i in base_huff:
		# print(i[0].ljust(10) + str(base_frequency[i[0]]).ljust(10) + i[1])
		base_code_length += base_frequency[i[0]]*len(i[1])
	print ("Base Huffman L :",base_code_length) #, "SH/BH",sample_sparse_code_length/base_code_length)
	return base_code_length
##########################################################


def encode(document, encodings):
	

	# print(encodings)
	doc_len = len(document)
	# encode_time_s = time.time()
	# encoded_doc = []
	encoded_doc = torch.zeros([30000000,],  dtype=torch.int8)
	count = 0
	# encoded_doc[count] = 123
	# count += 1
	for i in range(doc_len):
		# encoded_doc.append(encodings[document[i]])
		encd = encodings[str(int(document[i]))]
		# print("ENCD:: ",encd)
		for char in encd:
			encoded_doc[count] = ord(char)
			count += 1
		encoded_doc[count] = 61
		count += 1
	# # print("DOC_LEN", doc_len, count)
	# # encoded_doc  = "".join(encoded_doc)
	# encode_time_e = time.time()

	# end_time = time.time()
	# print("document",len(document),document[:10])
	# print("encoded_doc",count,encoded_doc[:25])
	return encoded_doc[:count]



# contaenates the document and frequencies and magnitude of quantisation
def get_tensor(swap_encodings, encoded_doc):
	# document, run_code_length, encodings, swap_encodings, encoded_doc  = run_length_huffman_core(quantized_grads,0, magnitude)
	# print("ENCODED DOC ::::::::::::\n",len(document), len(encoded_doc),document[-100:], "\n:::::::::::::::")
	le = 0
	for i in swap_encodings.keys():
		le = le + len(i) + len(swap_encodings[i])
	# print("q", len(quantized_grads)) 
	# print(run_code_length,  le)
	# print(swap_encodings)


	l = torch.zeros([150000,],  dtype=torch.int8)
	count = 0

	for key in swap_encodings.keys():
		for char in key:
			l[count] = ord(char)
			count += 1

		l[count] = 59 #ord(':')
		count += 1

		for char in swap_encodings[key]:
			l[count] = ord(char)
			count += 1
		
		l[count] = 60 #ord(':')
		count += 1

	l[count] = 61
	count += 1

	l = l[:count]
	# print(l[:count])
	# print(encoded_doc)
	third_tensor = torch.cat((l, encoded_doc), 0)
	# print(third_tensor.shape)
	return  third_tensor


def decode(padded_encoded_grad, size, magnitude, max_len=10000000):
	## Elements in padded_encoded_grad are valid only upto index size-1
	# print(len(padded_encoded_grad), padded_encoded_grad)
	idx = 0
	swap_encodings = {}
	key = ""
	c = int(padded_encoded_grad[idx])
	# print(c,padded_encoded_grad[idx] )
	idx+=1
	i=0
	while(c!=61 ):
		key=""
		while(c!=59):
			key+=chr(c) 
			c = int(padded_encoded_grad[idx])
			# print(c,padded_encoded_grad[idx])
			idx+=1
		val = ""
		c = padded_encoded_grad[idx]
		idx+=1
		while(c!=60):
			val+=chr(c) 
			c = int(padded_encoded_grad[idx])
			idx+=1
		# print(key, val)
		swap_encodings[key] = val
		c = int(padded_encoded_grad[idx])
		idx+=1
		i+=1

	doc = torch.zeros(max_len)
	check_doc = [None]*max_len
	check_id = 0
	d_id = 0
	c = int(padded_encoded_grad[idx])
	idx +=1 
  
	try:
		while(1):
			key=""
			while(c!= 61):
				key+= chr(c)
				c = int(padded_encoded_grad[idx])
				idx +=1
			
			
			check_doc[check_id] = swap_encodings[key]
			check_id+=1
			val = swap_encodings[key].split("c")
			if(int(val[0])==0):
				d_id += int(val[1])
			else:    
				for i in range(int(val[1])):
					doc[d_id] = int(val[0])
					d_id+= 1
			if(idx==size):
				# print("Exiting While")
				break
			c = int(padded_encoded_grad[idx])
			idx +=1
	
	except KeyError:
		
		print(key,idx,size)
		exit(0)
	return magnitude*doc[:d_id]
	# print("CHECK DOC ::::::::::::\n",check_id,check_doc[check_id-100:check_id+1],  "\n:::::::::::::::")
##decoded swap encodings
	# return swap_encodings
	# return check_doc

def decode_generic(padded_encoded_grad, size, magnitude,  max_len=10000000):
	## Elements in padded_encoded_grad are valid only upto index size-1
	# print(len(padded_encoded_grad), padded_encoded_grad)
	idx = 0
	swap_encodings = {}
	key = ""
	c = int(padded_encoded_grad[idx])
	# print(c,padded_encoded_grad[idx] )
	idx+=1
	i=0
	while(c!=61 ):
		key=""
		while(c!=59):
			key+=chr(c) 
			c = int(padded_encoded_grad[idx])
			# print(c,padded_encoded_grad[idx])
			idx+=1
		val = ""
		c = padded_encoded_grad[idx]
		idx+=1
		while(c!=60):
			val+=chr(c) 
			c = int(padded_encoded_grad[idx])
			idx+=1
		# print(key, val)
		swap_encodings[key] = val
		c = int(padded_encoded_grad[idx])
		idx+=1
		i+=1

#    print("decoded_swap_encodeing:\n",swap_encodings)
	doc = torch.zeros(max_len)
	check_doc = [None]*max_len
	check_id = 0
	d_id = 0
	c = int(padded_encoded_grad[idx])
	idx +=1 
  
	try:
		while(1):
			key=""
			while(c!= 61):
				key+= chr(c)
				c = int(padded_encoded_grad[idx])
				idx +=1
			
			
			check_doc[check_id] = swap_encodings[key]
			check_id+=1
			val = swap_encodings[key]
			doc[d_id] = int(val)
			d_id+= 1
			if(idx==size):
				# print("Exiting While")
				break
			c = int(padded_encoded_grad[idx])
			idx +=1
	
	except KeyError:
		
		print(key,idx,size)
		exit(0)
	return magnitude*doc[:d_id]
	# print("CHECK DOC ::::::::::::\n",check_id,check_doc[check_id-100:check_id+1],  "\n:::::::::::::::")

def decode_generic_sparse(padded_encoded_grad, size, magnitude,  max_len=50000000):
	## Elements in padded_encoded_grad are valid only upto index size-1
	# print(len(padded_encoded_grad), padded_encoded_grad)
	idx = 0
	swap_encodings = {}
	key = ""
	c = int(padded_encoded_grad[idx])
	# print(c,padded_encoded_grad[idx] )
	idx+=1
	i=0
	while(c!=61 ):
		key=""
		while(c!=59):
			key+=chr(c) 
			c = int(padded_encoded_grad[idx])
			# print(c,padded_encoded_grad[idx])
			idx+=1
		val = ""
		c = padded_encoded_grad[idx]
		idx+=1
		while(c!=60):
			val+=chr(c) 
			c = int(padded_encoded_grad[idx])
			idx+=1
		# print(key, val)
		swap_encodings[key] = val
		c = int(padded_encoded_grad[idx])
		idx+=1
		i+=1

	# print("decoded_swap_encodeing:\n",swap_encodings)
	doc = torch.zeros(max_len)
	check_doc = [None]*max_len
	check_id = 0
	d_id = 0
	c = int(padded_encoded_grad[idx])
	idx +=1 
  
	try:
		while(1):
			key=""
			while(c!= 61):
				key+= chr(c)
				c = int(padded_encoded_grad[idx])
				idx +=1
			
			
	#        check_doc[check_id] = swap_encodings[key]
	 #       check_id+=1
			val = swap_encodings[key]
			# print(val)
			if(val[0]=="0"):
				val = val.split("c")
				d_id += int(val[1])
			else:
				doc[d_id] = int(val)
				d_id+= 1
				if(idx==size):
					# print("Exiting While")
					break
			c = int(padded_encoded_grad[idx])
			idx +=1
	
	except KeyError:
		
		print("exiting",key, padded_encoded_grad[idx])#," ",idx," ",size)

		exit(0)
	return magnitude*doc[:d_id]
	# print("CHECK DOC ::::::::::::\n",check_id,check_doc[check_id-100:check_id+1],  "\n:::::::::::::::")

def decode_mapping(n):
	if (n%2==0):
		return int(-n/2)
	else:
		return int((n-1)/2)
	# return n

def decode_omega(encoded_doc, size, magnitude, max_len=10000000):
	
	# // can use memoization to store the values
	# swap_encodings = {}
	# print("Encoded",encoded_doc)
	decoded_doc = torch.zeros(max_len)
	doc_id = 0
	key = ""
	search_count = 0
	count = 0
	num = -256
	for i in range(size):
		key+= str(int(encoded_doc[i]))
		try:
			if(search_count ==0):
				l = decode_omega_coding(key)
				# print(i,l)
				# decoded_doc[doc_id] = decode_mapping(l)
				num = decode_mapping(l)
				# doc_id += 1
				# swap_encodings[key] = 
				key = ""
				search_count = 1
			else:
				l = decode_omega_coding(key)
				# print(i,l)
				count = l
				# doc_id += 1
				# swap_encodings[key] = 
				key = ""
				search_count=0
				
				for j in range(count):
					decoded_doc[doc_id] = num
					doc_id += 1

		except IndexError as e:
			pass

	if (key != ""):
		print("Error in decode_omega")
	# print(decoded_doc[:10])
	return decoded_doc[:doc_id]
