clc;
close all;
%% Read Dispersion Image 
fvi = 98; 
fvj = 701;
fid = fopen('FV.dat','r');
fv = fread(fid,[fvj fvi],'single');
fclose(fid);
%% 3D GMM
[nt, nx] = size(fv);
fvf=zeros(3,nt*nx);
g=1;
for i=1:nx
    for j=1:nt
        fvf(3,g)=fv(j,i);
        g=g+1;
        fvf(1,g)=i;
        fvf(2,g)=j;
    end
end
data(:,1) = fvf(1,:);
data(:,2) = fvf(2,:);
data(:,3) = fvf(3,:);

K = 2;
GMModel = fitgmdist(data,K);
T1 = cluster(GMModel,data);

cen=[mean(data(T1==1,:));mean(data(T1==2,:))];

dist=sum(cen.^2,2);
[dump,sortind]=sort(dist,'ascend');
newT1=zeros(size(T1));

for i =1:K
    newT1(T1==i)=find(sortind==i);
end

plotind=newT1==1;
data1=[data(plotind,1),data(plotind,2)];
%% 2D DBSCAN
epsilon = 4;
minpts = 45;
idx1 = DBSCAN(data1,epsilon,minpts);
%% Pick Max
pm = max(idx1);
for i=1:pm
    cn(i) = sum(idx1(:)==i);
end

data1_1=data1(:,1);
data1_2=data1(:,2);
k=1;
for i=1:pm
    if cn(i)>50
        cnn(k)=i;
        k=k+1;
    end
    a_even_logic = idx1==i;
    b1 = data1_1(a_even_logic);
    b2 = data1_2(a_even_logic);
    b = [b1 b2];
    name_string = ['s' num2str(i) '=b'];
    eval(name_string);
end

s1(1,1)=1;
s1(1,2)=1;
for i=1:pm
    tfv=zeros(nt,nx);
    ts=eval(['s',num2str(i)]);
    [tsm, tsn] = size(ts);
    for j=1:tsm
        ts1=ts(j,1);
        ts2=ts(j,2);
        tfv(ts2,ts1)=fv(ts2,ts1);
    end
    name_string = ['fv' num2str(i) '=tfv'];
    eval(name_string);
end

z = 1;
for i=1:pm
    tfv=eval(['fv',num2str(i)]);
    ts=eval(['s',num2str(i)]);
    [tsm, tsn] = size(ts);
    tsmin = min(ts(:,1));
    tsmax = max(ts(:,1));
    tdc=zeros(1,tsmax-tsmin+1);
    for j=tsmin:tsmax
        d=max(tfv(:,j));
        for k=1:nt
            if tfv(k,j)==d
                tdc(j)=k;
            end
        end
        if tdc(j)==nt&&i~=1&&i~=5
            tdc(j)=tdc(j-1);
        end
    end
    [~,np] = size(tdc);
    if np==fvi
        mode(z)=i;
        z=z+1;    
    end
    name_string = ['dc' num2str(i) '=tdc'];
    eval(name_string);
end
%% Particle Filter
d = 1;
k = 1;
n = nt;
A = 1;
G = eye(k)*1e-3;
C = 1;
S = eye(d)*1e-1;

mu0 = 0;
P0 = eye(k);

model.A = A;
model.G = G;
model.C = C;
model.S = S;
model.mu0 = mu0;
model.P0 = P0;

for i=mode
    tdc=eval(['dc',num2str(i)]);
    [row,col,tdc] = find(tdc);
    [m, n]= size(tdc);
    [tmodel, llh] = ldsEm(tdc,k);
    nu = PF(tmodel,tdc);
    tdcp = tmodel.C*nu;
    name_string = ['dcp' num2str(i) '=tdcp'];
    eval(name_string);
end
%% Figures
figure(1)
imagesc(fv);
colormap(jet);
axis xy;

figure(2)
surf(fv,'EdgeColor','None');
shading interp;
colormap(jet);
axis tight;
view([60,50]);

figure(3)
plot3(fvf(1,:),fvf(2,:),fvf(3,:),'.','MarkerSize',5,'color','k')
grid on
axis tight
view([60,50]);
xlabel('Phase Velocity(m/s)')
ylabel('Frequency(Hz)')
zlabel('Amplitude')
set(gca,'linewidth',3,'fontsize',10);

figure(4)
k=1;
for z=mode
    pf=eval(['fv',num2str(z)]);
    for i=1:nt
        for j=1:nx
            if pf(i,j)~=0
                fvi(i,j)=k;
            end
        end
    end
    k=k+1;
end
imagesc(fvi);
axis xy;
colormap(jet);

figure(5)
imagesc(fv);
hold on;
for i=mode
    pdc=eval(['dcp',num2str(i)]);
    ps=eval(['s',num2str(i)]);
    tsmin = min(ps(:,1));
    tsmax = max(ps(:,1));
    freq = (tsmin:tsmax);
    plot(freq,pdc,'.','MarkerSize',30,'Color','w');
    hold on;
    plot(freq,pdc,'-w','LineWidth',2.5);
end
axis xy;
colormap(jet);