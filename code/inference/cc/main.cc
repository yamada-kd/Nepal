#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <vector>
#include <random>
#include <unordered_map>
#include <regex>
#include <cstring>
#include <algorithm>
#include <sstream>
#define INF -999999
#include "cmdline.h"
using std::cout;
using std::endl;

void argsort(int *lii,float *lix,const int &n);
void csum(float &sum,const float *lix,const int &n);
void cssum(float &sum,const float *lix,const int &n);
std::vector<float> splitfloat(const std::string &str, char sep);
void trim(std::string& line, const char* trimCharacterList=" \t\v\r\n");
std::vector<std::string> splitstring(const std::string &str, char sep);
float max2(const float &x,const float &y);
float max3(const float &x,const float &y,const float &z);
void findmaxindex(int &pi,int &pj,float **H,int &maxi,int &maxj);
void findmaxvalueindexrow(float &rowmax,int &rowmaxindex,float **H,int &maxi,int &maxj);
void findmaxvalueindexcol(float &colmax,int &colmaxindex,float **H,int &maxi,int &maxj);
float align(std::vector<std::vector<float>>& lipssmi,std::vector<std::vector<float>>& lipssmj,Eigen::MatrixXf& w1,Eigen::MatrixXf& w2,float op,float ep,std::string mode,int UNIT,std::unordered_map<std::string,int> mapt);
float max3root(const float &x,const float &y,const float &z,int &root);
float f(Eigen::MatrixXf& w,const std::vector<std::vector<std::vector<float>>>& litrainx1part,const std::vector<std::vector<std::vector<float>>>& litrainx2part,const std::vector<std::unordered_map<std::string,int>>& litraintpart,const int UNIT,const float OP,const float EP,const std::string MODE,const int CPUN,const int BATCH);
void report(std::string outdir,std::string INPUT,int EPOCH,int BATCH,float OP,float EP,std::string MODE,int UNIT,float sigma,int TN,int VN,int mu,int SEED);
void getoption(cmdline::parser &a,int argc,char *argv[])
{
	a.add<int>("cpu",0,"The number of threads",true);
	a.set_program_name("dfnn");
	a.parse_check(argc,argv);
}

int main(int argc,char *argv[])
{
	int SEED=1;
	int UNIT=144;
	int BATCH=1536;
	float sigma=0.032;
	int TN=BATCH;
	int VN=160;
	int EPOCH=150;
	int n=(20+1)*UNIT+(UNIT+1)+1+1;
	int lambda=(int)((4+3*log(n))*2.5);
	int mu=(int)lambda/2;
	char INPUT[]="input/scop40_training.sp.txt";
	std::string MODE="sg";
	
	cmdline::parser option;
	getoption(option,argc,argv);
	std::string outdir=".";
	int CPUN=option.get<int>("cpu");
	float w_[n];
	{
		std::string weightfile="input/cc-half-144.txt";
		std::ifstream fin(weightfile.c_str());
		std::string line;
		while(getline(fin,line))
		{
			std::regex re(R"([\[\]])");
			line=std::regex_replace(line,re," ");
			trim(line);
			
			char del[]=" ";
			char *tok;
			char *cline=(char*)malloc(sizeof(char)*(line.length()+1));
			strcpy(cline,line.c_str());
			tok=strtok(cline,del);
			int i=0;
			while(tok!=NULL)
			{
				w_[i]=atof(tok);
				tok=strtok(NULL,del);
				i++;
			}
		}
	}
	Eigen::MatrixXf w(1,n);
	for(int i=0;i<n-2;i++)
	{
		w(0,i)=w_[i];
	}
//	w=Eigen::MatrixXf::Random(1,n);
	w(0,n-2)=-1.5;
	w(0,n-1)=-0.1;
	{
		w(0,n-2)=(-1)*fabs(w(0,n-2));
		w(0,n-1)=(-1)*fabs(w(0,n-1));
		float tmp=float(w(0,n-2));
		if(w(0,n-2)>w(0,n-1))
		{
			w(0,n-2)=w(0,n-1);
			w(0,n-1)=tmp;
		}
	}
	
	int TBATCHSIZE=(int)TN/BATCH;
	int VBATCHSIZE=(int)VN/BATCH;
	if(TBATCHSIZE<1)
	{
		TBATCHSIZE=1;
	}
	if(VBATCHSIZE<1)
	{
		VBATCHSIZE=1;
	}
	
	std::vector<std::vector<std::vector<float>>> litrainx1;
	std::vector<std::vector<std::vector<float>>> livalidx1;
	std::vector<std::vector<std::vector<float>>> litrainx2;
	std::vector<std::vector<std::vector<float>>> livalidx2;
	std::vector<std::unordered_map<std::string,int>> litraint;
	std::vector<std::unordered_map<std::string,int>> livalidt;
	{
		std::ifstream fin(INPUT);
		std::string line;
		int number=0;
		while(getline(fin,line))
		{
			trim(line);
			std::vector<std::string> tmp0=splitstring(line,';');
			std::unordered_map<std::string,int> mapt;
			std::vector<std::string> tmp1=splitstring(tmp0[5],' ');
			for(unsigned int i=0;i<tmp1.size();i++)
			{
				mapt[tmp1[i]]=1;
			}
			std::vector<std::vector<float>> lipssmx1;
			std::vector<std::string> tmp2=splitstring(tmp0[2],',');
			for(unsigned int i=0;i<tmp2.size();i++)
			{
				std::vector<float> liscore=splitfloat(tmp2[i],' ');
				lipssmx1.push_back(liscore);
			}
			std::vector<std::vector<float>> lipssmx2;
			std::vector<std::string> tmp3=splitstring(tmp0[4],',');
			for(unsigned int i=0;i<tmp3.size();i++)
			{
				std::vector<float> liscore=splitfloat(tmp3[i],' ');
				lipssmx2.push_back(liscore);
			}
			if(number<TN)
			{
				litraint.push_back(mapt);
				litrainx1.push_back(lipssmx1);
				litrainx2.push_back(lipssmx2);
			}
			else if(number<TN+VN)
			{
				livalidt.push_back(mapt);
				livalidx1.push_back(lipssmx1);
				livalidx2.push_back(lipssmx2);
			}
			number++;
		}
	}
	
	Eigen::MatrixXf C(n,n);
	C=Eigen::MatrixXf::Identity(n,n);
	Eigen::MatrixXf pc(1,n); 
	pc=Eigen::MatrixXf::Zero(1,n);
	Eigen::MatrixXf ps(1,n); 
	ps=Eigen::MatrixXf::Zero(1,n);
	float chin=sqrt(n)*(1-(float)1/(4*n)+(float)1/(21*n*n));
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(C);
	Eigen::MatrixXf diagD(1,n);
	Eigen::MatrixXf B(n,n);
	diagD=(es.eigenvalues()).transpose();
	B=es.eigenvectors();
	float tdiagD[n];
	for(int i=0;i<n;i++)
	{
		tdiagD[i]=diagD(i);
	}
	int indx[n];
	for(int i=0;i<n;i++)
	{
		indx[i]=i;
	}
	argsort(indx,tdiagD,n);
	for(int i=0;i<n;i++)
	{
		diagD(i)=pow(tdiagD[i],0.5);
	}
	Eigen::MatrixXf tB(n,n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			tB(i,j)=B(i,indx[j]);
		}
	}
	B=tB;
	Eigen::MatrixXf BD(n,n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			BD(i,j)=B(i,j)*diagD(j);
		}
	}
	float liweight[n];
	for(int i=0;i<mu;i++)
	{
		liweight[i]=log((float)mu+0.5)-log((float)i+1);
	}
	float sum=0;
	csum(sum,liweight,mu);
	for(int i=0;i<mu;i++)
	{
		liweight[i]=liweight[i]/sum;
	}
	float ssum=0;
	cssum(ssum,liweight,mu);
	Eigen::MatrixXf weight(1,mu);
	for(int i=0;i<mu;i++)
	{
		weight(0,i)=liweight[i];
	}
	float mueff=1/ssum;
	float cc=4/((float)n+4);
	float cs=(mueff+2)/(float)((float)n+mueff+3);
	float ccov1=2/(float)(pow(n+1.3,2)+mueff);
	float ccovmu=2*(mueff-2+1/mueff)/(pow(n+2,2)+mueff);
	if(ccovmu>1-ccov1)
	{
		ccovmu=1-ccov1;
	}
	float lidamp=pow((mueff-1)/(float)(n+1),0.5)-1;
	if(lidamp>0)
	{
		lidamp=1+2*lidamp+cs;
	}
	else
	{
		lidamp=1+cs;
	}
	
//	if(outdir!=".")
//	{
		report(outdir,INPUT,EPOCH,BATCH,float(w(0,n-2)),float(w(0,n-1)),MODE,UNIT,sigma,TN,VN,mu,SEED);
//	}
	
	/**************** learning loop ****************/
	std::random_device rnd;
	std::mt19937 mt(rnd());
	mt.seed(SEED);
	std::normal_distribution<> nd(0.0,1.0);
	std::ofstream fw(outdir+"/w.txt");
	for(int t=0;t<EPOCH;t++)
	{
		
		std::vector<int> liindex(TN);
		for(int i=0;i<TN;i++)
		{
			liindex[i]=i;
		}
		std::shuffle(std::begin(liindex),std::end(liindex),mt);
		float sumtraincost=0;
		for(int i=0;i<TBATCHSIZE;i++)
		{
			int start=i*BATCH;
			int end=start+BATCH;
			if(end>TN)
			{
				end=TN;
			}
			Eigen::MatrixXf old_w(1,n);
			old_w=w;
			Eigen::MatrixXf y(lambda,n);
			for(int i=0;i<lambda;i++)
			{
				for(int j=0;j<n;j++)
				{
					y(i,j)=nd(mt);
				}
			}
			Eigen::MatrixXf syBDT(lambda,n);
			syBDT=sigma*y*BD.transpose();
			for(int i=0;i<lambda;i++)
			{
				y.row(i)=old_w+syBDT.row(i);
			}
			
			for(int i=0;i<lambda;i++)
			{
				y(i,n-2)=(-1)*fabs(y(i,n-2));
				y(i,n-1)=(-1)*fabs(y(i,n-1));
				float tmp=float(y(i,n-2));
				if(y(i,n-2)>y(i,n-1))
				{
					y(i,n-2)=y(i,n-1);
					y(i,n-1)=tmp;
				}
			}
			
			std::unordered_map<int,float> mpfy; 
			for(int i=0;i<lambda;i++)
			{
				Eigen::MatrixXf w_now(1,n);
				w_now=y.row(i);
				std::vector<std::vector<std::vector<float>>> litrainx1part(BATCH);
				std::vector<std::vector<std::vector<float>>> litrainx2part(BATCH);
				std::vector<std::unordered_map<std::string,int>> litraintpart(BATCH);
				for(int j=0;j<BATCH;j++)
				{
					litrainx1part[j]=litrainx1[liindex[j]];
					litrainx2part[j]=litrainx2[liindex[j]];
					litraintpart[j]=litraint[liindex[j]];
				}
				mpfy[i]=f(w_now,litrainx1part,litrainx2part,litraintpart,UNIT,float(y(i,n-2)),float(y(i,n-1)),MODE,CPUN,BATCH);
			}
			
			int indx[lambda];
			float val[lambda];
			int c=0;
			for(auto itr=mpfy.begin();itr!=mpfy.end();itr++)
			{
				indx[c]=(int)itr->first;
				val[c]=(float)itr->second;
				c++;
			}
			argsort(indx,val,lambda);
			float tmptraincost=0;
			for(int i=0;i<mu;i++)
			{
				tmptraincost+=val[i];
			}
			sumtraincost+=tmptraincost/mu;
			Eigen::MatrixXf population(mu,n);
			for(int i=0;i<mu;i++)
			{
				population.row(i)=y.row(indx[i]);
			}
			w=weight*population;
			Eigen::MatrixXf c_diff(1,n);
			c_diff=w-old_w;
			for(int i=0;i<n;i++)
			{
				diagD(i)=1/diagD(i);
			}
			Eigen::MatrixXf c_diffBT(1,n);
			c_diffBT=c_diff*B.transpose();
			Eigen::MatrixXf diagDB(1,n);
			diagDB=diagD*B;
			for(int i=0;i<n;i++)
			{
				ps(i)=(1-cs)*ps(i)+pow(cs*(2-cs)*mueff,0.5)/sigma*diagDB(i)*c_diffBT(i);
			}
			
			float hsig=ps.norm()/pow(1.0-pow(1.0-cs,2*(t+1)),0.5)/chin;
			if(hsig<1.4+2/(n+1))
			{
				hsig=1.0;
			}
			else
			{
				hsig=0.0;
			}
			for(int i=0;i<n;i++)
			{
				pc(i)=(1-cc)*pc(i)+pow(cc*(2-cc)*mueff,0.5)/sigma*c_diff(i);
			}
			Eigen::MatrixXf artmp(mu,n);
			for(int i=0;i<mu;i++)
			{
				artmp.row(i)=population.row(i)-old_w;
			}
			Eigen::MatrixXf weightartmpT(n,mu);
			for(int i=0;i<n;i++)
			{
				for(int j=0;j<mu;j++)
				{
					weightartmpT(i,j)=weight(j)*artmp.transpose()(i,j);
				}
			}
			Eigen::MatrixXf weightartmpTartmp(n,n);
			weightartmpTartmp=weightartmpT*artmp;
			Eigen::MatrixXf outerpcpc(n,n);
			for(int i=0;i<n;i++)
			{
				for(int j=0;j<n;j++)
				{
					outerpcpc(i,j)=pc(0,i)*pc(0,j);
				}
			}
			for(int i=0;i<n;i++)
			{
				for(int j=0;j<n;j++)
				{
					C(i,j)=(1-ccov1-ccovmu+(1-hsig)*ccov1*cc*(2-cc))*C(i,j)+ccov1*outerpcpc(i,j)+ccovmu*weightartmpTartmp(i,j)/pow(sigma,2);
				}
			}
			sigma=sigma*exp((ps.norm()/chin-1)*cs/lidamp);
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(C);
			diagD=(es.eigenvalues()).transpose();
			B=es.eigenvectors();
			for(int i=0;i<n;i++)
			{
				indx[i]=i;
			}
			for(int i=0;i<n;i++)
			{
				tdiagD[i]=diagD(0,i);
			}
			argsort(indx,tdiagD,n);
			for(int i=0;i<n;i++)
			{
				diagD(i)=pow(tdiagD[i],0.5);
			}
			for(int i=0;i<n;i++)
			{
				for(int j=0;j<n;j++)
				{
					tB(i,j)=B(i,indx[j]);
				}
			}
			B=tB;
			for(int i=0;i<n;i++)
			{
				for(int j=0;j<n;j++)
				{
					BD(i,j)=B(i,j)*diagD(j);
				}
			}
		}
		std::ofstream fw(outdir+"/w.txt",std::ios::app);
		fw<<w<<endl;
		float traincost=(float)sumtraincost/TBATCHSIZE;
		float trainacc=1-traincost;
		char message[100];
		sprintf(message,"Epoch %6d: Training acc=%7.4f (Sigma=%7.4f)\n",t+1,trainacc,sigma);
		std::string smessage=message;
		if(outdir==".")
		{
			cout<<smessage;
		}
		
		if((t+1)>=50 && (t+1)%5==0)
		{
			float validcost=f(w,livalidx1,livalidx2,livalidt,UNIT,float(w(0,n-2)),float(w(0,n-1)),MODE,CPUN,VN);
			sprintf(message,"              Validation acc=%7.4f\n",1-validcost);
			smessage=message;
			if(outdir==".")
			{
				cout<<smessage;
			}
		}
	}
}

void argsort(int *lii,float *lix,const int &n)
{
	float tmpx;
	int tmpi;
	for(int i=0;i<n;i++)
	{
		for(int j=i+1;j<n;j++)
		{
			tmpi=lii[i];
			tmpx=lix[i];
			if(lix[i]>lix[j])
			{
				lix[i]=lix[j];
				lix[j]=tmpx;
				lii[i]=lii[j];
				lii[j]=tmpi;
			}
		}
	}
}
void csum(float &sum,const float *lix,const int &n)
{
	for(int i=0;i<n;i++)
	{
		sum+=lix[i];
	}
}
void cssum(float &sum,const float *lix,const int &n)
{
	for(int i=0;i<n;i++)
	{
		sum+=pow(lix[i],2);
	}
}
void trim(std::string& line, const char* trimCharacterList)
{
	std::string::size_type left=line.find_first_not_of(trimCharacterList);
	if (left!=std::string::npos)
	{
		std::string::size_type right=line.find_last_not_of(trimCharacterList);
		line=line.substr(left,right-left+1);
	}
}
std::vector<float> splitfloat(const std::string &str, char sep)
{
	std::vector<float> v;
	std::stringstream ss(str);
	std::string buffer;
	while(std::getline(ss,buffer,sep))
	{
		float a;
		a=stof(buffer);
		v.push_back(a);
	}
	return v;
}
std::vector<std::string> splitstring(const std::string &str, char sep)
{
	std::vector<std::string> v;
	std::stringstream ss(str);
	std::string buffer;
	while(std::getline(ss,buffer,sep))
	{
		v.push_back(buffer);
	}
	return v;
}
void findmaxvalueindexcol(float &colmax,int &colmaxindex,float **H,int &maxi,int &maxj)
{
	colmax=INF;
	for(int i=0;i<maxi+1;i++)
	{
		if(colmax<H[i][maxj])
		{
			colmax=H[i][maxj];
			colmaxindex=i;
		}
	}
}
void findmaxvalueindexrow(float &rowmax,int &rowmaxindex,float **H,int &maxi,int &maxj)
{
	rowmax=INF;
	for(int j=0;j<maxj+1;j++)
	{
		if(rowmax<H[maxi][j])
		{
			rowmax=H[maxi][j];
			rowmaxindex=j;
		}
	}
}
void findmaxindex(int &pi,int &pj,float **H,int &maxi,int &maxj)
{
	int max=INF;
	for(int i=0;i<maxi+1;i++)
	{
		for(int j=0;j<maxj+1;j++)
		{
			if(max<H[i][j])
			{
				max=H[i][j];
				pi=i;
				pj=j;
			}
		}
	}
}
float max2(const float &x,const float &y)
{
	if(x>y)
	{
		return x;
	}
	else
	{
		return y;
	}
}
float max3(const float &x,const float &y,const float &z)
{
	float max=x;
	float tmp[2]={y,z};
	for(int i=0;i<2;i++)
	{
		if(max<tmp[i])
		{
			max=tmp[i];
		}
	}
	return max;
}
float max3root(const float &x,const float &y,const float &z,int &root)
{
	float max=x; // DIAG, root=DIAG
	if(max<y)
	{
		max=y;
		root=1; // LEFT
	}
	if(max<z)
	{
		max=z;
		root=2; // UP
	}
	return max;
}
float align(std::vector<std::vector<float>>& lipssmi,std::vector<std::vector<float>>& lipssmj,Eigen::MatrixXf& w1,Eigen::MatrixXf& w2,float op,float ep,std::string mode,int UNIT,std::unordered_map<std::string,int> mapt)
{
	int NONE=0,LEFT=1,UP=2,DIAG=3;
	int maxi=lipssmi.size();
	int maxj=lipssmj.size();
	
	float **H=(float**)malloc(sizeof(float*)*(maxi+1));
	for(int i=0;i<maxi+1;i++)
	{
		H[i]=(float*)malloc(sizeof(float)*(maxj+1));
	}
	for(int i=0;i<maxi+1;i++)
	{
		for(int j=0;j<maxj+1;j++)
		{
			H[i][j]=0;
		}
	}
	float **I=(float**)malloc(sizeof(float*)*(maxi+1));
	for(int i=0;i<maxi+1;i++)
	{
		I[i]=(float*)malloc(sizeof(float)*(maxj+1));
	}
	for(int i=0;i<maxi+1;i++)
	{
		for(int j=0;j<maxj+1;j++)
		{
			I[i][j]=INF;
		}
	}
	float **J=(float**)malloc(sizeof(float*)*(maxi+1));
	for(int i=0;i<maxi+1;i++)
	{
		J[i]=(float*)malloc(sizeof(float)*(maxj+1));
	}
	for(int i=0;i<maxi+1;i++)
	{
		for(int j=0;j<maxj+1;j++)
		{
			J[i][j]=INF;
		}
	}
	int **P=(int**)malloc(sizeof(int*)*(maxi+1));
	for(int i=0;i<maxi+1;i++)
	{
		P[i]=(int*)malloc(sizeof(int)*(maxj+1));
	}
	for(int i=0;i<maxi+1;i++)
	{
		for(int j=0;j<maxj+1;j++)
		{
			P[i][j]=0;
		}
	}
	
	if(mode=="sg" ||mode=="nw")
	{
		for(int i=0;i<maxi+1;i++)
		{
			for(int j=0;j<maxj+1;j++)
			{
				if(i==0 && j==0)
				{
					P[i][j]=NONE;
				}
				else if(i==0)
				{
					P[i][j]=LEFT;
				}
				else if(j==0)
				{
					P[i][j]=UP;
				}
			}
		}
		if(mode=="nw")
		{
			for(int i=0;i<maxi+1;i++)
			{
				for(int j=0;j<maxj+1;j++)
				{
					if(i==0 && j==0)
					{
						H[i][j]=0;
					}
					else if(i==0)
					{
						H[i][j]=op+(j-1)*ep;
					}
					else if(j==0)
					{
						H[i][j]=op+(i-1)*ep;
					}
				}
			}
		}
	}
	
	Eigen::MatrixXf l1(1,40+1);
	Eigen::MatrixXf l2o(1,UNIT+1);
	Eigen::MatrixXf l2i(1,UNIT);
	int pi=0,pj=0;
	for(int i=1;i<maxi+1;i++)
	{
		for(int a=0;a<20;a++)
		{
			l1(0,a)=lipssmi[i-1][a];
		}
		for(int j=1;j<maxj+1;j++)
		{
			for(int a=0;a<20;a++)
			{
				l1(0,a+20)=lipssmj[j-1][a];
			}
			l1(0,40)=1;
			l2i=l1*w1;
			for(int a=0;a<UNIT;a++)
			{
				if(l2i(0,a)<=0)
				{
					l2o(0,a)=0;
				}
				else
				{
					l2o(0,a)=l2i(0,a);
				}
			}
			l2o(0,UNIT)=1;
			float similarity=(float)(l2o*w2)(0,0);
			I[i][j]=max2(H[i-1][j]+op,I[i-1][j]+ep);
			J[i][j]=max2(H[i][j-1]+op,J[i][j-1]+ep);
			float diagscore=max3(H[i-1][j-1],I[i-1][j-1],J[i-1][j-1])+similarity;
			float leftscore=J[i][j];
			float upscore=I[i][j];
			int root=DIAG;
			float maxscore=max3root(diagscore,leftscore,upscore,root);
			
			H[i][j]=maxscore;
			if(mode=="sw")
			{
				H[i][j]=max2(0,maxscore);
			}
			if(mode=="sw")
			{
				if(H[i][j]==0)
				{
					/* Nothing to do */
				}
				else
				{
					P[i][j]=root;
				}
			}
			else
			{
				P[i][j]=root;
			}
			pj=j;
		}
		pi=i;
	}
	
	if(mode=="sw")
	{
		findmaxindex(pi,pj,H,maxi,maxj);
	}
	else if(mode=="sg")
	{
		float rowmax=0;
		int rowmaxindex=0;
		findmaxvalueindexrow(rowmax,rowmaxindex,H,maxi,maxj);
		float colmax=0;
		int colmaxindex=0;
		findmaxvalueindexcol(colmax,colmaxindex,H,maxi,maxj);
		if(rowmax>colmax)
		{
			for(int i=0;i<maxj+1;i++)
			{
				if(i>rowmaxindex)
				{
					P[maxi][i]=LEFT;
				}
			}
		}
		else
		{
			for(int i=0;i<maxi+1;i++)
			{
				if(i>colmaxindex)
				{
					P[i][maxj]=UP;
				}
			}
		}
	}
	
	int pv=P[pi][pj];
	std::vector<std::string> alseqir;
	std::vector<std::string> alseqjr;
	std::string gapstring="-";
	while(pv!=NONE)
	{
		if(pv==DIAG)
		{
			pi--;
			pj--;
			alseqir.push_back(std::to_string(pi));
			alseqjr.push_back(std::to_string(pj));
		}
		else if(pv==LEFT)
		{
			pj--;
			alseqir.push_back(gapstring);
			alseqjr.push_back(std::to_string(pj));
		}
		else if(pv==UP)
		{
			pi--;
			alseqir.push_back(std::to_string(pi));
			alseqjr.push_back(gapstring);
		}
		pv=P[pi][pj];
	}
	
	for(int i=0;i<maxi+1;i++)
	{
		free(H[i]);
	}
	free(H);
	for(int i=0;i<maxi+1;i++)
	{
		free(I[i]);
	}
	free(I);
	for(int i=0;i<maxi+1;i++)
	{
		free(J[i]);
	}
	free(J);
	for(int i=0;i<maxi+1;i++)
	{
		free(P[i]);
	}
	free(P);
	
	/* scoring */
	int count=0;
	for(unsigned int a=0;a<alseqir.size();a++)
	{
		if(mapt.count(alseqir[a]+"-"+alseqjr[a])!=0)
		{
			count++;
		}
	}
	
	float acc=(float)count/mapt.size();
	float cost=1-acc;
	return cost;
}
float f(Eigen::MatrixXf& w,const std::vector<std::vector<std::vector<float>>>& litrainx1part,const std::vector<std::vector<std::vector<float>>>& litrainx2part,const std::vector<std::unordered_map<std::string,int>>& litraintpart,const int UNIT,const float OP,const float EP,const std::string MODE,const int CPUN,const int BATCH)
{
	Eigen::MatrixXf w1(40+1,UNIT);
	Eigen::MatrixXf w2(UNIT+1,1);
	int m=-1,n=0;
	for(int i=0;i<(20)*UNIT;i++)
	{
		if(i%UNIT==0)
		{
			m++;
			n=0;
		}
		w1(m,n)=w(0,i);
		w1(m+20,n)=w(0,i);
		n++;
	}
	n=0;
	for(int i=(20)*UNIT;i<(20+1)*UNIT;i++)
	{
		w1(40,n)=w(0,i);
		n++;
	}
	for(int i=0;i<UNIT+1;i++)
	{
		w2(i,0)=w(0,i+(20+1)*UNIT);
	}
	
	float liscore[BATCH];
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(CPUN)
	#endif
	for(int i=0;i<BATCH;i++)
	{
		std::vector<std::vector<float>> lipssmi=litrainx1part[i];
		std::vector<std::vector<float>> lipssmj=litrainx2part[i];
		std::unordered_map<std::string,int> mapt=litraintpart[i];
		float score=align(lipssmi,lipssmj,w1,w2,OP,EP,MODE,UNIT,mapt);
		liscore[i]=score;
	}
	
	float sum=0;
	for(int i=0;i<BATCH;i++)
	{
		sum+=liscore[i];
	}
	
	float meancost=sum/BATCH;
	return meancost;
}
void report(std::string outdir,std::string INPUT,int EPOCH,int BATCH,float OP,float EP,std::string MODE,int UNIT,float sigma,int TN,int VN,int mu,int SEED)
{
	std::ofstream fout(outdir+"/log.txt");
	fout<<"INPUT        : "<<INPUT<<endl;
	fout<<"EPOCH        : "<<EPOCH<<endl;
	fout<<"TN           : "<<TN<<endl;
	fout<<"VN           : "<<VN<<endl;
	fout<<"BATCH        : "<<BATCH<<endl;
	fout<<"OP           : "<<OP<<endl;
	fout<<"EP           : "<<EP<<endl;
	fout<<"MODE         : "<<MODE<<endl;
	fout<<"UNIT         : "<<UNIT<<endl;
	fout<<"SIGMA        : "<<sigma<<endl;
	fout<<"MU           : "<<mu<<endl;
	fout<<"SEED         : "<<SEED<<endl;
	fout<<endl;
}
