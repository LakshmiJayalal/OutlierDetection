clear all
close all
%-------------- Get data--------------------------
D=double(image_fn(1,1));
% -------------- Subspace recovery -------------------------------


% -------------- Algorithm 1 Subspace recovery using isearch -------------
[N,c]=size(D);
    Cstar=zeros(N,c);
beta1 =0.5;
alpha=1.2;
invHtH= inv(D*D' + beta1*eye(size(D,1)));
obj_val=zeros(1,c);
for i = 1:c
     disp(i)
    [chat,~] = admm(D,(D(:,i))',invHtH,1000,1.2);% 
%     chat = basis_pursuit(Pinv_D,D,D(:,i)',1,rho1,alpha1);
    Cstar(:,i)=chat;
    obj_val(i)=norm(chat'*D,1);
   
end
%Step 2: Direction search
x=1./vecnorm(D'*Cstar,1);
figure;plot(x);
%% Subspace Recovery
figure
hist(x,287)
r=3;%Verify
[~,ind]=sort(x);
ss=orth(D(:,ind(1:r)));
leakage_metric=vecnorm(D*(eye(c)-ss*ss'))./vecnorm(D);
figure;plot(leakage_metric);
%% Modified GramSchmidt
[metric outlier_index U flag]= gram_schmidt_final(x,D);