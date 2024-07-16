clear;clc;close all;
addpath('multi_datasets')
addpath('funs')

dataset_name='pre_Yale';
load([dataset_name,'.mat'])
c=length(unique(Y));
V=length(X);
N=size(X{1},1);

%% Parameter Setting of pre_Yale          ACC = 0.75
gamma=1400;
filter=2;

%% Parameter Setting of pre_ORL           ACC = 0.77
% gamma=2400;
% filter=2;

%% Parameter Setting of 100leaves         ACC = 0.90
% gamma=1000;
% filter=3;

%% Optimization of MvC-DBGF
[S,obj,W,alpha,beta] = solution_MvC_DBGF(X,V,c,N,gamma,filter);
S(S<1e-5)=0;
[clusternum1, y_learned]=graphconncomp(sparse(S));
final = y_learned';
result = ClusteringMeasure_new(Y,final);

disp(['********************************************']);
disp(['Running MvC-DBGF on ',dataset_name,' to obtain ACC: ', num2str(result.ACC)]);
disp(['********************************************']);



