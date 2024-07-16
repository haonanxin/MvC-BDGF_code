function [S,obj,W,alpha,beta] = solution_MvC_DBGF(X,V,c,N,eta,index)

%% ===================== Initialization =====================

alpha = orth(rand(V,1));
beta = orth(rand(V,1));
lambda=2^10;
iter=50;
zr=1e-11;

A=cell(V,1);
f_A=cell(V,1);
f_AX=cell(V,1);
SS=zeros(N,N);
d=zeros(V,1);

for i=1:V
    d(i)=size(X{i},2);
    max_val1=max(max(X{i}));
    min_val1=min(min(X{i}));
    X{i}=(X{i}-min_val1)./(max_val1-min_val1);
    A{i,1}=constructW_PKN(X{i},5);

    SS=SS+A{i,1};
    [filter_X] = all_filters(X{i},A{i,1},index);
    f_AX{i,1}=filter_X;
end

SS=(SS+SS')/2;
L=diag(sum(SS,2))-SS+eye(N)*eps;
[F, temp, ev]=eig1(L,c+1, 0);
F=F(:,2:c+1);
F = F./repmat(sqrt(sum(F.^2,2)),1,c);

S=SS./V;

sum_dim=sum(d);
reduced_d=ceil(sum_dim*0.1);
W0 = orth(rand(sum_dim,reduced_d));
obj=zeros(iter,1);
%% =====================  updating =====================
for t=1:iter
    % update alpha
    beta_A=0;
    for i=1:V
        beta_A=beta_A+beta(i)*A{i,1};
    end
    U=S'*beta_A*S;

    W=cell(V,1);
    count=1;
    for i=1:V
        W{i,1}=W0(count:count+d(i)-1,:);
        count=count+d(i);
    end
    AA=cell(V,1);
    BB=cell(V,1);
    AA_ba=zeros(N*reduced_d,V);
    BB_ba=zeros(N*reduced_d,V);
    for i=1:V
        BB{i,1}=f_AX{i,1}*W{i,1};
        BB_ba(:,i)=reshape(BB{i,1},[],1);
        AA{i,1}=U*BB{i,1};
        AA_ba(:,i)=reshape(AA{i,1},[],1);
    end
    AABB=AA_ba'*BB_ba;
    alpha = eig1(AABB,1,1);

    % update beta
    alpha_fa_X_W=zeros(N,reduced_d);
    for i=1:V
        alpha_fa_X_W=alpha_fa_X_W+alpha(i)*BB{i,1};
    end
    temp1=S*alpha_fa_X_W;
    temp2=temp1*temp1';
    C=zeros(V,1);
    for i=1:V
        C(i)=trace(A{i,1}*temp2);
    end
    beta=C./sqrt(sum(C.^2));

    % update W
    beta_A=0;
    for i=1:V
        beta_A=beta_A+beta(i)*A{i,1};
    end
    U=S'*beta_A*S;
    X_ba=[];
    for i=1:V
        X_ba=[X_ba,alpha(i)*f_AX{i,1}];
    end
    W0=eig1(X_ba'*U*X_ba,reduced_d,1);
    W=cell(V,1);
    count=1;
    for i=1:V
        W{i,1}=W0(count:count+d(i)-1,:);
        count=count+d(i);
    end

    % update S
    P=0;
    Q=0;
    for i=1:V
        P=P+alpha(i)*f_AX{i,1}*W{i,1};
        Q=Q+beta(i)*A{i,1};
    end
    distance= pdist(F);
    K=lambda*squareform(distance.^2);
    PPT=P*P';
    [S] = update_S_QP_new(S,PPT,K,Q,N,eta);

    % updata F
    S=(S+S')/2;
    L=diag(sum(S,2))-S+eye(N)*eps;
    F_old=F;
    [F, temp3, ev]=eig1(L,c, 0);
    obj(t)=-trace(S*PPT*S'*Q)+eta*trace(S'*S)+lambda*trace(F'*L*F);
    fprintf('iter:%d\n',t);

    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > zr
        lambda = 2*lambda;
    elseif fn2 < zr
        lambda = lambda/2;
        F = F_old;
    else
        break;
    end

end


end


function [filter_X] = all_filters(X,A,index)
N=size(A,1);
% A=full(A);
A=A+eye(N);
A=(A'+A)/2;
D=diag(sum(A,2))^(-0.5);
L=eye(N)-D*A*D;
[eigvec, eigval] = eigs(L, N);
val=diag(eigval);
switch index
    case 1         % all-pass
        f_A=eye(N);
        filter_X=f_A*X;
    case 2         % low-pass
        re_lambda=1 - 0.5*val;
        f_A=eigvec*(diag(re_lambda))*(eigvec)^(-1);
        filter_X=f_A*X;
    case 3         % high-pass
        re_lambda=0.5*val;
        f_A=eigvec*(diag(re_lambda))*(eigvec)^(-1);
        filter_X=f_A*X;
    case 4         % band-pass
        re_lambda=exp(-10*(val-1).^2);
        f_A=eigvec*(diag(re_lambda))*(eigvec)^(-1);
        filter_X=f_A*X;
    case 5         % band-reject
        re_lambda=1-exp(-10*(val-1).^2);
        f_A=eigvec*(diag(re_lambda))*(eigvec)^(-1);
        filter_X=f_A*X;
    case 6         % comb
        re_lambda=abs(sin(pi*val));
        f_A=eigvec*(diag(re_lambda))*(eigvec)^(-1);
        filter_X=f_A*X;
    case 7         % low-band-pass
        re_lambda=1.*(val>0. & val<=0.5)+(exp(-100*(val-0.5).^2)).*(val>0.5 & val<=1)+(exp(-50*(val-1.5).^2)).*(val>1 & val<=2);
        f_A=eigvec*(diag(re_lambda))*(eigvec)^(-1);
        filter_X=f_A*X;
    case 8         % high-band-pass
        re_lambda=1.*(val>1.5 & val<=2)+(exp(-100*(val-1.5).^2)).*(val>1 & val<=1.5)+(exp(-50*(val-0.5).^2)).*(val>0 & val<=1);
        f_A=eigvec*(diag(re_lambda))*(eigvec)^(-1);
        filter_X=f_A*X;
end
end