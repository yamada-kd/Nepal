#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,re

def main():
	parser=argparse.ArgumentParser(description="Generating paramter.h from the result of CMA-ES.")
	parser.add_argument('-i',type=str,dest="input",required=True,help='Weight file.')
	parser.add_argument('-e',type=int,dest="epoch",required=True,help='The number of learning epoch.')
	args=parser.parse_args()
	UNIT,w1,w2,op,ep=readweight(args.input,args.epoch)
	
	print("#ifndef _PARAMETER_H_")
	print("#define _PARAMETER_H_")
	print()
	print("float op=%(op)s;"%locals())
	print("float ep=%(ep)s;"%locals())
	print("float w1[41][%(UNIT)s]="%locals())
	print("{")
	for i in range(len(w1)):
		print("\t","{",end="",sep="")
		for j in range(len(w1[i])):
			print("{0:13.5e}".format(w1[i][j]),sep="",end="")
			if j<len(w1[i])-1: print(",",end="")
		if i<len(w1)-1:
			print("},")
		else:
			print("}")
	print("};")
	UNITp1=UNIT+1
	print("float w2[%(UNITp1)s][1]="%locals())
	print("{")
	for i in range(len(w2)):
		print("\t","{",end="",sep="")
		for j in range(len(w2[i])):
			print("{0:13.5e}".format(w2[i][j]),sep="",end="")
			if j<len(w2[i])-1: print(",",end="")
		if i<len(w2)-1:
			print("},")
		else:
			print("}")
	print("};")
	print()
	print("#endif")

def readweight(fweight,epoch):
	fin=open(fweight,"r")
	liw=[]
	for i,line in enumerate(fin):
		if i+1==epoch:
			line=line.replace("]","").replace("[","")
			line=line.strip()
			litmp=re.split("\s+",line)
			liw=[float(tmp) for tmp in litmp]
	unit=len(liw)//22
	gap=len(liw)%22
	op,ep=0,0
	if gap==3:
		op=liw[-2]
		ep=liw[-1]
	else:
		op=-1.5
		ep=-0.1
	w1,w2=[[0 for j in range(unit)] for i in range(40+1)],[[0 for j in range(1)] for i in range(unit+1)]
	m,n=-1,0
	for i in range(20*unit):
		if i%unit==0:
			m+=1
			n=0
		w1[m][n]=liw[i];
		w1[m+20][n]=liw[i];
		n+=1
	n=0
	for i in range(20*unit,(20+1)*unit):
		w1[40][n]=liw[i]
		n+=1
	for i in range(unit+1):
		w2[i][0]=liw[i+(20+1)*unit]
	return unit,w1,w2,op,ep


if __name__ == '__main__':
	main()
