#ifndef _STRUCT_H_
#define _STRUCT_H_

struct inputdata
{
	std::string name;
	std::string seq;
	std::vector<std::vector<float>> pssm;
	std::string alseq="";
};

#endif
