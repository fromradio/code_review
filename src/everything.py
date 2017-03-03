#!/usr/bin/env python
# Author: add your name
# Func: add function

import os
import random
import sys
import math
import numpy as np
import cv2

def partition(work_dir, all_dir, ratio):
	f = open(work_dir + 'all.txt', 'w')
	label_value = 0
	label_name = []
	filelist=[]
	length=[]
	imgslist=[]

	dirs=os.listdir(all_dir)

	for dir_ in dirs:
		if os.path.isdir(os.path.join(all_dir,dir_)):
			imgslist=[]
			class_dir=os.path.join(all_dir,dir_)
			imgs_dirs=os.listdir(class_dir)
			for imgs_dir_ in imgs_dirs:
				if os.path.isdir(os.path.join(class_dir,imgs_dir_)):
					imgs_dir=os.path.join(class_dir,imgs_dir_)
					for fname in os.listdir(imgs_dir):
						relative_path=dir_+'/'+imgs_dir_+'/'+fname
						print>>f,relative_path,label_value
						imgslist.append(relative_path)
			label_name.append(dir_)
			filelist.append(imgslist)
			length.append(len(filelist[label_value]))
			label_value+=1

	f.close()

	cross_len=sys.maxint
	for l in length:
		if l < cross_len:
			cross_len=l
	deci = cross_len/10*ratio
	cross_len = int(deci*10)
	cross_dir = work_dir + 'x_validate/'
	if os.path.exists(cross_dir)==False:
		os.mkdir(cross_dir)
	for i in range(10):
    		test=open(cross_dir+'test'+str(i)+'.txt','w+')
    		train=open(cross_dir+'train'+str(i)+'.txt','w+')
    		test_=[]
    		train_=[]
	        for count in range(cross_len):
			label_value=0
			if i*deci <= count and count < (i+1)*deci:
				for name in label_name:
					test_.append([filelist[label_value][count],str(label_value)])
					label_value+=1
			else:
				for name in label_name:
					train_.append([filelist[label_value][count],str(label_value)])
					label_value+=1
		random.shuffle(test_)
		random.shuffle(train_)
		for j in test_:
			test.write(j[0]+' '+j[1]+'\n')
		for j in train_:
			train.write(j[0]+' '+j[1]+'\n')
		test.close()
		train.close()
def _partition(work_dir,all_dir,ratio):
	f=open(work_dir+'all.txt','w')
	label_value=0
	label_name=[]
	filelist=[]
	length=[]
	for root,sub_dirs,nu1 in os.walk(all_dir):
		for sub_dir in sub_dirs:
			filelist.append([])
			filelist[label_value]=os.listdir(root+sub_dir)
			label_name.append(sub_dir)
			length.append(len(filelist[label_value]))
			for fname in filelist[label_value]:
				print>>f,sub_dir+'/'+fname,label_value
			label_value+=1

	f.close()

	cross_len=sys.maxint
	for l in length:
		if l < cross_len:
			cross_len=l
	deci = int(math.ceil(ratio*cross_len/10))
	cross_len = deci*10
	cross_dir = work_dir + 'x_validate/'
	if os.path.exists(cross_dir)==False:
		os.mkdir(cross_dir)
	for i in range(10):
    		test=open(cross_dir+'test'+str(i)+'.txt','w+')
    		train=open(cross_dir+'train'+str(i)+'.txt','w+')
    		test_=[]
    		train_=[]
	        for count in range(cross_len):
			label_value=0
			if i*deci <= count and count < (i+1)*deci:
				for name in label_name:
					test_.append([name+'/'+filelist[label_value][count],str(label_value)])
					label_value+=1
			else:
				for name in label_name:
					train_.append([name+'/'+filelist[label_value][count],str(label_value)])
					label_value+=1
		random.shuffle(test_)
		random.shuffle(train_)
		for j in test_:
			test.write(j[0]+' '+j[1]+'\n')
		for j in train_:
			train.write(j[0]+' '+j[1]+'\n')
		test.close()
		train.close()

def split_by_name(imgs_dir):

	filelist=os.listdir(imgs_dir)

	for file_ in filelist:
		temp=file_.split('_')
		if len(temp)<3:
			continue
		subdir=os.path.join(imgs_dir,temp[1])
		if os.path.exists(subdir)==False:
			os.mkdir(subdir)
		full_path=os.path.join(imgs_dir,file_)
		os.system('mv '+full_path+' '+subdir)

def input_labeled_filenames(imgs_dir,label,input_filename):
	input_file=open(input_filename,'a')
	files=os.listdir(imgs_dir)
	for file_ in files:
		full_name=os.path.join(imgs_dir,file_)
		print>>input_file,full_name,label

def construct_cross_validate(all_dir,all_file,work_dir):
    all_reader=open(all_file,'r')
    labels=[]
    filelist=[]
    lines=all_reader.readlines()
    for line in lines:
        label=line.strip('\n').split(' ')[-1]
        if label not in labels:
            labels.append(label)
            filelist.append([])
        filelist[labels.index(label)].append(line.strip('\n'))

    counts=np.zeros([len(filelist)],np.int)
    for i in range(len(filelist)):
        counts[i]=len(filelist[i])
    cross_len=int(counts.min())
    deci=cross_len/10
    cross_len=deci*10

    cross_dir=os.path.join(work_dir,'x_validate/')
    if os.path.exists(cross_dir)==False:
        os.mkdir(cross_dir)

    for i in range(10):
        test=open(cross_dir+'test'+str(i)+'.txt','w+')
        train=open(cross_dir+'train'+str(i)+'.txt','w+')
        test_=[]
        train_=[]
        for count in range(cross_len):
            if i*deci <= count and count < (i+1)*deci:
                for some_class in filelist:
                    test_.append(some_class[count])
            else:
                for some_class in filelist:
                    train_.append(some_class[count])
        random.shuffle(test_)
        random.shuffle(train_)
        for j in test_:
            print>>test,j
        for j in train_:
            print>>train,j
        test.close()
        train.close()

    all_reader.close()
    return len(labels)

def input_data_by_fname(all_dir, tr_fn, te_fn, pixel_len, color_channel, label_size):
    train_file=open(tr_fn,'r')
    test_file=open(te_fn,'r')

    lines=train_file.readlines()

    X_tr=np.zeros((len(lines),pixel_len,pixel_len,color_channel), np.float16)
    Y_tr=np.zeros((len(lines),label_size))

    index=0
    for line in lines:
        filename=line.strip('\n').split(' ')[0]
        label_value=int(line.strip('\n').split(' ')[1])
        img=cv2.imread(all_dir+filename)
        b,g,r=cv2.split(img)
        X_tr[index,:,:,0]=r
        X_tr[index,:,:,1]=g
        X_tr[index,:,:,2]=b
        label_vec=np.zeros(label_size)
        label_vec[label_value]=1
        Y_tr[index]=label_vec

        index+=1

    lines=test_file.readlines()

    X_te=np.zeros((len(lines),pixel_len,pixel_len,color_channel), np.float16)
    Y_te=np.zeros((len(lines),label_size))

    index=0
    for line in lines:
        filename=line.strip('\n').split(' ')[0]
        label_value=int(line.strip('\n').split(' ')[1])
        img=cv2.imread(all_dir+filename)
        b,g,r=cv2.split(img)
        X_te[index,:,:,0]=r
        X_te[index,:,:,1]=g
        X_te[index,:,:,2]=b
        label_vec=np.zeros(label_size)
        label_vec[label_value]=1
        Y_te[index]=label_vec

        index+=1

    return X_tr, Y_tr, X_te, Y_te

def input_data(imgs_dir, pixel_len, color_channel):
	filelist=os.listdir(imgs_dir)
	imgs_num=len(filelist)
	imgs_data=np.zeros([imgs_num, pixel_len, pixel_len, color_channel])
	index=0
	for file_ in filelist:
		img=cv2.imread(os.path.join(imgs_dir,file_))
		b,g,r=cv2.split(img)
		imgs_data[index,:,:,0]=r
		imgs_data[index,:,:,1]=g
		imgs_data[index,:,:,2]=b
		index+=1
	return imgs_data

def minus_mean(X, mean_image):
    for channel in range(X.shape[3]):
        for elm in range(X.shape[0]):
            X[elm, :, :, channel]-=mean_image[:,:,channel]

def cal_true_class(teY):
    true_class=np.zeros(teY.shape[0])
    for i in range(true_class.shape[0]):
        true_class[i]=np.argmax(teY[i])
    return true_class
