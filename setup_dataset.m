function [] = setup_dataset(baseDir)

% baseDir = 'C:\Users\sebas\Documents\MATLAB\DataProCiencia\DeepLearning\';
rawDir = fullfile(baseDir,'raw');
refDir = fullfile(baseDir,'ref');
trainDir = fullfile(baseDir,'train');
testDir = fullfile(baseDir,'test');

[~, ~, ~] = mkdir(trainDir);
[~, ~, ~] = mkdir(fullfile(trainDir,'input'));
[~, ~, ~] = mkdir(fullfile(trainDir,'output'));
[~, ~, ~] = mkdir(testDir);
[~, ~, ~] = mkdir(fullfile(testDir,'input'));
[~, ~, ~] = mkdir(fullfile(testDir,'output'));

files = dir(fullfile(rawDir,'*.mat'));

%%
blocksize = 10;     % Block size in wavelengths
freq_L = 2e6; freq_H = 10e6;
freq_C = mean([freq_L freq_H]);
deltaF = 100E3;

overlap_pc      = 0.8;
ratio_zx        = 1;

x_inf = -2; x_sup = 2; 
z_inf = 0.05; z_sup = 4;
test_pc = 20;
% dynRange = [-40,0];

%%
iFile = 1;
load(fullfile(files(iFile).folder,files(iFile).name));

dx = x(2)-x(1);
dz = z(2)-z(1);
sam1 = rf;
x = x*100; z = z*100;

% Limits for ACS estimation
ind_x = x_inf <= x & x <= x_sup;
ind_z = z_inf <= z & z <= z_sup;
x = x(ind_x);
z = z(ind_z);
sam1 = sam1(ind_z,ind_x);


% Wavelength size
c0 = 1540;
wl = c0/freq_C;   % Wavelength (m)

% Lateral samples
wx = round(blocksize*wl*(1-overlap_pc)/dx);  % Between windows
nx = round(blocksize*wl/dx);                 % Window size
x0 = 1:wx:length(x)-nx;
x_ACS = x(1,x0+round(nx/2));
n  = length(x0);

% Axial samples
wz = round(blocksize*wl*(1-overlap_pc)/dz * ratio_zx); % Between windows
nz = 2*round(blocksize*wl/dz /2 * ratio_zx); % Window size
L = (nz/2)*dz*100;   % (cm)
z0p = 1:wz:length(z)-nz;
z0d = z0p + nz/2;
z_ACS = z(z0p+ nz/2);
m  = length(z0p);

NFFT = round(fs/deltaF);
band = (0:NFFT-1)'/NFFT * fs;   % [Hz] Band of frequencies
rang = band > freq_L & band < freq_H ;   % useful frequency range
f  = band(rang)*1e-6; % [MHz]
p = length(f);

%% Generating Diffraction compensation
% Generating references
referenceAtt = 0.5;
att_ref = referenceAtt*(f.^1.05)/(20*log10(exp(1)));
att_ref_map = zeros(m,n,p);
for jj=1:n
    for ii=1:m
        att_ref_map(ii,jj,:) = att_ref;
    end
end

% Windows for spectrum
% windowing = tukeywin(nz/2,0.25);
windowing = hamming(nz/2);
windowing = windowing*ones(1,nx);

% For looping
refFiles = dir([refDir,'\*.mat']);
Nref = length(refFiles);
% swrap = saran_wrap(band); % Correction factor for phantom data
swrap = 0;
% Memory allocation
Sp_ref = zeros(m,n,p,Nref);
Sd_ref = zeros(m,n,p,Nref);
for iRef = 1:Nref
    out = load([refDir,'\',refFiles(iRef).name]);
    samRef = out.rf;
    samRef = samRef(ind_z,ind_x); % Cropping
    % figure,imagesc(db(hilbert(samRef)))
    for jj=1:n
        for ii=1:m
            xw = x0(jj) ;   % x window
            zp = z0p(ii);
            zd = z0d(ii);

            sub_block_p = samRef(zp:zp+nz/2-1,xw:xw+nx-1);
            sub_block_d = samRef(zd:zd+nz/2-1,xw:xw+nx-1);
            [tempSp,~] = spectra(sub_block_p,windowing,swrap,nz/2,NFFT);
            [tempSd,~] = spectra(sub_block_d,windowing,swrap,nz/2,NFFT);

            Sp_ref(ii,jj,:,iRef) = (tempSp(rang));
            Sd_ref(ii,jj,:,iRef) = (tempSd(rang));
        end
    end
end

Sp = mean(Sp_ref,4); Sd = mean(Sd_ref,4);
compensation = ( log(Sp) - log(Sd) ) - 4*L*att_ref_map;

% Liberating memory to avoid killing my RAM
clear Sp_ref Sd_ref

%%
rng(14);
files = files(randperm(length(files)));
num_test = round(length(files)*test_pc/100);

x_offset = x(1);
[X,Z] = meshgrid(x, z);

Sp = zeros(m,n,p);
Sd = zeros(m,n,p);
acs = zeros(m,n);

for iFile = 1:length(files)
    tic
    out = load(fullfile(files(iFile).folder,files(iFile).name));
    sam1 = out.rf(ind_z,ind_x);
    
    % Ideal attenuation
    cx = center_meters(1) * 100 + x_offset;
    cz = center_meters(2) * 100;
    rx = radius_meters(1) * 100;
    rz = radius_meters(2) * 100;
    inc = ((X - cx)/rx).^2 + ((Z - cz)/rz).^2 < 1;
    att_ideal = ones(size(X))*alpha_mean(1);
    att_ideal(inc) = alpha_mean(2);
    
    % Spectrum and block ACS
    for jj=1:n
        for ii=1:m
            xw = x0(jj) ;   % x window
            zp = z0p(ii);
            zd = z0d(ii);
    
            sub_block_p = sam1(zp:zp+nz/2-1,xw:xw+nx-1);
            sub_block_d = sam1(zd:zd+nz/2-1,xw:xw+nx-1);
    
            block_acs = att_ideal(zp:zp+nz-1,xw:xw+nx-1);
    
            [tempSp,~] = spectra(sub_block_p,windowing,0,nz/2,NFFT);
            [tempSd,~] = spectra(sub_block_d,windowing,0,nz/2,NFFT);
            Sp(ii,jj,:) = (tempSp(rang));
            Sd(ii,jj,:) = (tempSd(rang));
            acs(ii,jj) = mean(block_acs(:));
        end
    end
    sld = (log(Sp) - log(Sd)) - (compensation);
    spectrum = (log(Sp) + log(Sd))/2;
    
    file_name = files(iFile).name;
    if iFile <= num_test
        save(fullfile(testDir,'input',file_name),'sld','spectrum');
        save(fullfile(testDir,'output',file_name),'acs','x_ACS','z_ACS');
    else
        save(fullfile(trainDir,'input',file_name),'sld','spectrum');
        save(fullfile(trainDir,'output',file_name),'acs','x_ACS','z_ACS');
    end
    toc
end

end

function [spect,psnr_Sp]=spectra(block,windowing,saran_layer,~,NFFT)
% Computes the average of the parallel echoes spectra in the ROI
%
% Inputs:
%   block           Data matrix, size nw x nx
%   windowing       Window vector, size nw x nx (the same for all cols)
%   saran_layer     Vector containing spectrum correction for the saran
%                   layer, of size NFFT. 0 is case there is no correction.
%   nw              Axial length of block (deprecated)
%   NFFT            Number of FFT points
%
% Outputs:
%   spect           Average of the parallel echoes spcetrin the ROI
%   psnr_Sp         Log spectrum
%       

block = block - mean(block);
block = block.*windowing;

spect = abs(fft(block,NFFT,1));     % Fourier transform proximal window
spect = spect.^2;                   % Sp is Intensity Now 

spect = mean(spect,2);   % Sp is the averaga of the parallel echoes in the ROI


% Saran-wrap correction factor for phantoms
if all(saran_layer)
    spect = spect./saran_layer;
end

% psnr_Sp = 10*log10(max(spect)./spect(end/2,:)); 
psnr_Sp = mean(log(spect),2);
end