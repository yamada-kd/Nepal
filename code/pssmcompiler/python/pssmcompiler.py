#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,re,os,uuid,subprocess

def main():
	parser=argparse.ArgumentParser(description="This program converts a fasta file to Nepal readable PSSM text file.")
	parser.add_argument("-i",type=str,dest="input",required=True,help='')
	parser.add_argument("-d",type=str,dest="db",required=False,default="../pseudodb/pseudodb",help="")
	parser.add_argument("-r",type=str,dest="rpsdb",required=False,default="~/db/cdd_delta/cdd_delta",help="")
	parser.add_argument("-b",type=str,dest="binary",required=False,default="~/local/bin/deltablast",help="")
	args=parser.parse_args()
	db,rpsdb,deltablast=args.db,args.rpsdb,args.binary
	liname,liseq=readfasta(args.input)
	dipssm,diseq={},{}
	for i in range(len(liname)):
		tmpseq="/tmp/seq."+str(uuid.uuid4())
		fout=open(tmpseq,"w")
		print(">",liname[i],"\n",liseq[i],"\n",sep="",end="",file=fout)
		fout.close()
		tmp="/tmp/"+str(uuid.uuid4())
		subprocess.call("%(deltablast)s -num_alignments 0 -num_iterations 1 -query %(tmpseq)s -rpsdb %(rpsdb)s -db %(db)s -out_ascii_pssm %(tmp)s > /dev/null"%locals(),shell=True)
		fin=open(tmp,"r")
		seq,lipssm="",[]
		for line in fin:
			line=line.strip()
			if re.search("^[0-9]",line):
				litmp=re.split("\s+",line)
				seq+=litmp[1]
				lipssm.append(" ".join(litmp[2:]))
		fin.close()
		os.remove(tmp)
		os.remove(tmpseq)
		if seq==liseq[i]:
			print("<name>",liname[i],sep="")
			print("<sequence>",liseq[i],sep="")
		strpssm=",".join(lipssm)
		print("<pssm>",strpssm,sep="")

def readfasta(input):
	liname,liseq=[],[]
	fin=open("%(input)s"%locals(),'r')
	tmpline=""
	for line in fin:
		line=line.rstrip()
		if line.startswith('>'):
			line=line.replace(">","")
			liname.append(line)
			if tmpline:
				liseq.append(tmpline)
				tmpline=""
		else:
			tmpline+=line.upper()
	liseq.append(tmpline)
	fin.close()
	return [liname,liseq]

if __name__ == '__main__':
	main()
