function [model, llh] = ldsEm(X, m)
if isstruct(m)   % init with a model
    model = m;
elseif numel(m) == 1  % random init with latent dimension m
    model = init(X,m);
end
tol = 1e-4;
maxIter = 2000;
llh = -inf(1,maxIter);
for iter = 2:maxIter
%     E-step
    [nu, U, llh(iter),Ezz, Ezy] = PF(model,X);
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
%     M-step 
    model = maximization(X, nu, U, Ezz, Ezy);
end
llh = llh(2:iter);

function model = init(X, k)
% d = size(X,1);
% model.mu0 = randn(k,1);
% model.P0 = iwishrnd(eye(k),k);
% model.A = randn(k,k);
% model.G = iwishrnd(eye(k),k);
% model.C = randn(d,k);
% model.S = iwishrnd(eye(d),d);
[A,C,Z] = ldsPca(X,k,3*k);
model.mu0 = Z(:,1);
E = Z(:,1:end-1)-Z(:,2:end);
model.P0 = (dot(E(:),E(:))/(k*size(E,2)))*eye(k);
model.A = A;
E = A*Z(:,1:end-1)-Z(:,2:end);
model.G = E*E'/size(E,2);
model.C = C;
E = C*Z-X(:,1:size(Z,2));
model.S = E*E'/size(E,2);

function model = maximization(X ,nu, U, Ezz, Ezy)
n = size(X,2);

EZZ = sum(Ezz,3);
EZY = sum(Ezy,3);
A = EZY/(EZZ-Ezz(:,:,n));                         
G = (EZZ-Ezz(:,:,1)-EZY*A')/(n-1);                

Xnu = X*nu';
C = Xnu/EZZ;                                      
S = (X*X'-Xnu*C')/n;                              

model.mu0 = nu(:,1);                              
model.P0 = U(:,:,1);                              
model.A = A;
model.G = (G+G')/2;
model.C = C;
model.S = (S+S')/2;