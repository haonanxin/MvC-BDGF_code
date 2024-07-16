function [S] = update_S_QP_new(S,B,K,A,num,beta)

NE=100;
obj=zeros(NE,1);
T=B*S';
for t=1:NE
    S_old=S;
    S=zeros(num,num);
    for i=1:num
        AD_i=A(:,i);
        AD_i(i)=0;
        Z_i=T*AD_i;
        H=2*(beta*eye(num)-A(t,t)*B);
        f=K(i,:)'-Z_i;

        x=EProjSimplex_new(-f/(2*beta));
        S(i,:)=x';
        T(:,i)=B*x;
    end
    obj(t)=-trace(S*B*S'*A)+trace(K'*S)+beta*trace(S'*S);
    if t>3&&abs(obj(t)-obj(t-1))<abs(0.001*obj(t))
        break
    end
end
end