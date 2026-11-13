clear; close all; clc; rng(20261112,'twister');
folder = '/Users/me/Documents/my simulation/jojo';
out = fullfile(folder,'CFC_movie_results','未命名文件夹');
if ~exist(out,'dir'), mkdir(out); end
Fs = 1000; nfft = 2048; nPerm = 2000;
low = 4:30; high = 30:5:120;
time = -.2:.001:.5; targets = -.2:.05:.5;
contrasts = {'0.00625','0.025','0.1'};
names = {'Contrast 0.00625','Contrast 0.025','Contrast 0.1','All contrasts pooled'};
C = cell(3,2,27,19); PX = cell(3,2,27); PY = cell(3,2,27,19);
centers = cell(1,27); counts = zeros(3,2);

%% 1. Trial-level spectral products from original channels 15 and 24
for ic = 1:3
    for g = 1:2
        file = dir(fullfile(folder,sprintf('*err%d_ind_contra%s_ERP_epoch_NA.mat',7*(g-1),contrasts{ic})));
        assert(numel(file)==1,'Expected exactly one raw input file');
        S = load(fullfile(file.folder,file.name));
        a = S.epoch_NA;
        x = squeeze(a(15,:,:)).'; y = squeeze(a(24,:,:)).';
        clear S a;
        counts(ic,g) = size(x,1);
        t = -2:1/Fs:(-2+(size(x,2)-1)/Fs);
        crop = find(t>=-1 & t<=.5); tc=t(crop); x=x(:,crop);
        X = cell(1,27); windows=cell(1,27); kernels=cell(1,27);
        for il = 1:27
            k=round(3*Fs/low(il)); k=k+double(mod(k,2)==0);
            starts=1:max(1,round(k*.1)):numel(crop)-k+1;
            windows{il}=starts(:)+(0:k-1);
            centers{il}=mean(tc(windows{il}),2).';
            bin=round(low(il)*nfft/Fs);
            kernels{il}=hann(k,'periodic').*exp(-1i*2*pi*bin*(0:k-1)'/nfft);
            z=zeros(size(x,1),numel(starts),'like',1i);
            for w=1:numel(starts)
                z(:,w)=x(:,windows{il}(w,:))*kernels{il};
            end
            X{il}=single(z); PX{ic,g,il}=abs(double(X{il})).^2;
        end
        for ih=1:19
            m=round(5*Fs/high(ih)); m=m+double(mod(m,2)==0);
            kernel=hann(m,'periodic').*exp(-1i*2*pi*high(ih)*(0:m-1)'/Fs);
            power=single(abs(conv2(y,kernel(end:-1:1).','same')/Fs).^2);
            power=power(:,crop);
            for il=1:27
                z=zeros(size(x,1),size(windows{il},1),'like',1i);
                for w=1:size(z,2)
                    z(:,w)=double(power(:,windows{il}(w,:)))*kernels{il};
                end
                z=double(single(z)); xx=double(X{il});
                C{ic,g,il,ih}=xx.*conj(z);
                PY{ic,g,il,ih}=abs(z).^2;
            end
        end
        fprintf('Raw coefficients: contrast %s, err%d, n=%d\n',contrasts{ic},7*(g-1),counts(ic,g));
    end
end
clear x y power X z xx;

%% 2. Original-group and shuffled-group CFC; baseline division each time
W=cell(3,1);
for ic=1:3
    W{ic}=zeros(nPerm,sum(counts(ic,:)));
    for p=1:nPerm
        order=randperm(sum(counts(ic,:)));
        W{ic}(p,order(1:counts(ic,1)))=1;
    end
end
R=nan(27,19,numel(time),2,4); pval=nan(27,19,numel(targets),4);
actual=nan(27,numel(targets));
for il=1:27
    wt=centers{il}; base=wt>=-.2 & wt<=0;
    [distance,idx]=min(abs(wt(:)-targets),[],1);
    valid=distance<=.02500001; used=[];
    for k=1:numel(idx)
        if any(used==idx(k)), valid(k)=false;
        elseif valid(k), used(end+1)=idx(k); end 
    end
    actual(il,valid)=wt(idx(valid));
    for ih=1:19
        obs=cell(4,2,3); perm=cell(4,2,3);
        for ic=1:3
            for j=1:3
                if j==1, a=double(C{ic,1,il,ih}); b=double(C{ic,2,il,ih});
                elseif j==2, a=double(PX{ic,1,il}); b=double(PX{ic,2,il});
                else, a=double(PY{ic,1,il,ih}); b=double(PY{ic,2,il,ih}); end
                obs{ic,1,j}=sum(a,1); obs{ic,2,j}=sum(b,1);
                perm{ic,1,j}=W{ic}*[a;b];
                perm{ic,2,j}=sum([a;b],1)-perm{ic,1,j};
            end
        end
        % Pool trials across contrasts brfore CFC and baseline normalization.
        for g=1:2
            for j=1:3
                obs{4,g,j}=obs{1,g,j}+obs{2,g,j}+obs{3,g,j};
                perm{4,g,j}=perm{1,g,j}+perm{2,g,j}+perm{3,g,j};
            end
        end
        for row=1:4
            ro=cell(1,2); rp=cell(1,2);
            for g=1:2
                c=abs(obs{row,g,1})./sqrt(obs{row,g,2}.*obs{row,g,3});
                ro{g}=c/mean(c(base));
                R(il,ih,:,g,row)=reshape(interp1(wt,ro{g},time,'linear',NaN),1,1,[]);
                c=abs(perm{row,g,1})./sqrt(perm{row,g,2}.*perm{row,g,3});
                rp{g}=c./mean(c(:,base),2);
            end
            d=ro{1}-ro{2}; null=rp{1}-rp{2}; mu=mean(null,1);
            p=(1+sum(abs(null-mu)>=abs(d-mu),1))/(nPerm+1);
            pval(il,ih,valid,row)=reshape(p(idx(valid)),1,1,[]);
        end
    end
    fprintf('Permutation: low frequency %d Hz\n',low(il));
end
% BH-FDR: all four rows, frequencies and tested times form one family.
Q=nan(size(pval)); ok=isfinite(pval); p=pval(ok); [v,order]=sort(p);
v=v.*numel(p)./(1:numel(p))'; v=flipud(cummin(flipud(v)));
q=zeros(size(p)); q(order)=min(v,1); Q(ok)=q;
D=reshape(R(:,:,:,1,:)-R(:,:,:,2,:),27,19,numel(time),4);
save(fullfile(out,'jojo_relative_CFC_results.mat'),'R','D','Q','pval','actual', ...
    'time','low','high','counts','centers','nPerm','targets','-v7.3');

%% 3. Movies
blueWhiteRed=interp1([0 .5 1],[.02 .19 .38;1 1 1;.40 0 .12],linspace(0,1,256));
for fdr=0:1
    if fdr
        filename='jojo_relative_CFC_FDR.mp4'; frames=1:10:numel(time);
    else
        filename='jojo_relative_CFC_12maps.mp4'; frames=1:numel(time);
    end
    fig=figure('Color','w','Position',[50 50 1500 1200]);
    tiledlayout(4,3,'TileSpacing','compact','Padding','compact'); h=gobjects(4,3);
    for row=1:4
        for col=1:3
            ax=nexttile; h(row,col)=imagesc(high,low,nan(27,19)); axis xy;
            set(ax,'Color',[.90 .91 .92]);
            xlabel('High-frequency power (Hz)'); ylabel('Low frequency (Hz)'); colorbar;
            if col<3
                clim([0 4]); colormap(ax,turbo);
                title(sprintf('%s | err%d',names{row},7*(col-1)));
            else
                clim([-3 3]); colormap(ax,blueWhiteRed);
                title([names{row} ' | err0 - err7']);
            end
        end
    end
    v=VideoWriter(fullfile(out,filename),'MPEG-4'); v.FrameRate=6; v.Quality=95; open(v);
    for it=frames
        for row=1:4
            for col=1:2
                z=R(:,:,it,col,row); set(h(row,col),'CData',z,'AlphaData',double(isfinite(z)));
            end
            z=D(:,:,it,row); alpha=double(isfinite(z));
            if fdr
                alpha=.14*alpha;
                for il=1:27
                    dist=abs(actual(il,:)-time(it)); dist(~isfinite(dist))=Inf;
                    [d,k]=min(dist);
                    if d<=.02500001, alpha(il,Q(il,:,k,row)<.05)=1; end
                end
                alpha(~isfinite(z))=0;
            end
            set(h(row,3),'CData',z,'AlphaData',alpha);
        end
        if fdr, label='FDR corrected CFC difference'; else, label='Relative CFC'; end
        sgtitle(sprintf('JOJO | %s | Time = %.0f ms',label,1000*time(it)));
        drawnow; F=getframe(fig); rgb=F.cdata;
        writeVideo(v,rgb(1:2*floor(size(rgb,1)/2),1:2*floor(size(rgb,2)/2),:));
    end
    close(v); close(fig);
end
