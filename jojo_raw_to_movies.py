"""Jojo raw MAT -> Relative CFC -> condition permutations/BH-FDR -> two movies.
Install: python3 -m pip install numpy scipy matplotlib imageio-ffmpeg
Run:     python3 jojo_raw_to_movies.py
This is a standalone script with fixed paths, matching jojo_raw_to_movies.m.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import convolve, windows as signal_windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio_ffmpeg

folder = Path('/Users/me/Documents/my simulation/jojo')
out = folder / 'CFC_movie_results' / '未命名文件夹'
out.mkdir(parents=True, exist_ok=True)
Fs, nfft, nPerm = 1000, 2048, 2000
low, high = np.arange(4, 31), np.arange(30, 121, 5)
time = np.linspace(-.2, .5, 701)
targets = np.linspace(-.2, .5, 15)
contrasts = ['0.00625', '0.025', '0.1']
names = ['Contrast 0.00625', 'Contrast 0.025', 'Contrast 0.1', 'All contrasts pooled']
rng = np.random.default_rng(20261112)
C, PX, PY = {}, {}, {}
centers = [None] * 27
counts = np.zeros((3, 2), dtype=int)

# 1. Trial-level spectral products from MATLAB channels 15 and 24.
for ic, contrast in enumerate(contrasts):
    for g, err in enumerate([0, 7]):
        files = list(folder.glob(f'*err{err}_ind_contra{contrast}_ERP_epoch_NA.mat'))
        assert len(files) == 1, f'Expected one raw MAT file: {contrast}, err{err}'
        S = loadmat(files[0], variable_names=['epoch_NA'])
        a = S['epoch_NA']  # [channel, time, trial]; Python indexes start at 0.
        x, y = a[14].T.copy(), a[23].T.copy()
        del S, a
        counts[ic, g] = x.shape[0]
        t = -2 + np.arange(x.shape[1]) / Fs
        crop = np.flatnonzero((t >= -1) & (t <= .5))
        tc, x = t[crop], x[:, crop]
        X, windows, kernels = [], [], []
        for il, f in enumerate(low):
            k = int(np.floor(3 * Fs / f + .5))
            k += int(k % 2 == 0)
            step = max(1, int(np.floor(k * .1 + .5)))
            starts = np.arange(0, len(crop) - k + 1, step)
            win = starts[:, None] + np.arange(k)
            wt = np.mean(tc[win], axis=1)
            if centers[il] is not None:
                np.testing.assert_allclose(wt, centers[il], rtol=0, atol=1e-12)
            centers[il] = wt
            fft_bin = int(np.floor(f * nfft / Fs + .5))
            kernel = signal_windows.hann(k, sym=False) * np.exp(-2j * np.pi * fft_bin * np.arange(k) / nfft)
            z = np.column_stack([x[:, w] @ kernel for w in win]).astype(np.complex64)
            X.append(z)
            PX[ic, g, il] = abs(z.astype(np.complex128)) ** 2
            windows.append(win)
            kernels.append(kernel)
        for ih, f in enumerate(high):
            m = int(np.floor(5 * Fs / f + .5))
            m += int(m % 2 == 0)
            kernel = signal_windows.hann(m, sym=False) * np.exp(-2j * np.pi * f * np.arange(m) / Fs)
            power = (abs(convolve(y, kernel[::-1][None, :], mode='same', method='fft') / Fs) ** 2).astype(np.float32)
            power = power[:, crop]
            for il in range(27):
                z = np.column_stack([power[:, w].astype(float) @ kernels[il] for w in windows[il]])
                z = z.astype(np.complex64).astype(np.complex128)
                xx = X[il].astype(np.complex128)
                C[ic, g, il, ih] = xx * z.conj()
                PY[ic, g, il, ih] = abs(z) ** 2
        print(f'Raw coefficients: contrast {contrast}, err{err}, n={counts[ic, g]}', flush=True)
        del x, y, power, X, z, xx

# 2. Original and shuffled groups; baseline is recomputed after each shuffle.
W = []
for n0, n1 in counts:
    w = np.zeros((nPerm, n0 + n1))
    for p in range(nPerm):
        w[p, rng.permutation(n0 + n1)[:n0]] = 1
    W.append(w)
R = np.full((27, 19, len(time), 2, 4), np.nan)
pval = np.full((27, 19, len(targets), 4), np.nan)
actual = np.full((27, len(targets)), np.nan)
for il, f in enumerate(low):
    wt = centers[il]
    base = (wt >= -.2) & (wt <= 0)
    assert base.any(), 'No baseline windows'
    idx = np.argmin(abs(wt[:, None] - targets), axis=0)
    valid = abs(wt[idx] - targets) <= .02500001
    used = set()
    for k, index in enumerate(idx):
        if index in used:
            valid[k] = False
        elif valid[k]:
            used.add(index)
    actual[il, valid] = wt[idx[valid]]
    for ih in range(19):
        obs, perm = {}, {}
        for ic in range(3):
            for j in range(3):
                if j == 0:
                    a, b = C[ic, 0, il, ih], C[ic, 1, il, ih]
                elif j == 1:
                    a, b = PX[ic, 0, il], PX[ic, 1, il]
                else:
                    a, b = PY[ic, 0, il, ih], PY[ic, 1, il, ih]
                pooled = np.concatenate([a, b])
                obs[ic, 0, j], obs[ic, 1, j] = a.sum(0), b.sum(0)
                perm[ic, 0, j] = W[ic] @ pooled
                perm[ic, 1, j] = pooled.sum(0) - perm[ic, 0, j]
        for g in range(2):
            for j in range(3):
                obs[3, g, j] = sum(obs[c, g, j] for c in range(3))
                perm[3, g, j] = sum(perm[c, g, j] for c in range(3))
        for row in range(4):
            ro, rp = [], []
            for g in range(2):
                c = abs(obs[row, g, 0]) / np.sqrt(obs[row, g, 1] * obs[row, g, 2])
                assert np.isfinite(c).all() and c[base].mean() > 0
                ro.append(c / c[base].mean())
                R[il, ih, :, g, row] = np.interp(time, wt, ro[g], left=np.nan, right=np.nan)
                c = abs(perm[row, g, 0]) / np.sqrt(perm[row, g, 1] * perm[row, g, 2])
                assert np.isfinite(c).all() and np.all(c[:, base].mean(axis=1) > 0)
                rp.append(c / c[:, base].mean(axis=1, keepdims=True))
            d, null = ro[0] - ro[1], rp[0] - rp[1]
            mu = null.mean(0)
            p = (1 + np.sum(abs(null - mu) >= abs(d - mu), axis=0)) / (nPerm + 1)
            pval[il, ih, valid, row] = p[idx[valid]]
    print(f'Permutation: low frequency {f} Hz', flush=True)

# BH-FDR: four contrasts x all tested frequencies/times, one family.
Q = np.full_like(pval, np.nan)
ok = np.isfinite(pval)
p = pval[ok]
order = np.argsort(p)
v = p[order] * len(p) / np.arange(1, len(p) + 1)
v = np.minimum.accumulate(v[::-1])[::-1]
q = np.empty_like(p)
q[order] = np.minimum(v, 1)
Q[ok] = q
D = R[:, :, :, 0, :] - R[:, :, :, 1, :]
centers_mat = np.empty((1, 27), dtype=object)
for il in range(27):
    centers_mat[0, il] = centers[il][None, :]
savemat(out / 'jojo_relative_CFC_results.mat', dict(R=R, D=D, Q=Q, pval=pval,
        actual=actual, time=time, low=low, high=high, counts=counts,
        centers=centers_mat, nPerm=nPerm, targets=targets), do_compression=True)

# 3. Ordinary and FDR movies.
blue_white_red = LinearSegmentedColormap.from_list('BWR', [[.02, .19, .38], [1, 1, 1], [.40, 0, .12]])
for fdr in [False, True]:
    filename = 'jojo_relative_CFC_FDR.mp4' if fdr else 'jojo_relative_CFC_12maps.mp4'
    frames = range(0, len(time), 10 if fdr else 1)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), dpi=100)
    fig.subplots_adjust(left=.065, right=.95, bottom=.07, top=.92, hspace=.5, wspace=.3)
    images = []
    for row in range(4):
        for col in range(3):
            ax = axes[row, col]
            ax.set_facecolor((.9, .91, .92))
            image = ax.imshow(np.full((27, 19), np.nan), origin='lower', aspect='auto',
                extent=[27.5, 122.5, 3.5, 30.5], interpolation='nearest',
                cmap='turbo' if col < 2 else blue_white_red,
                vmin=0 if col < 2 else -3, vmax=4 if col < 2 else 3)
            ax.set_xlabel('High-frequency power (Hz)', fontsize=9)
            ax.set_ylabel('Low frequency (Hz)', fontsize=9)
            ax.set_title(names[row] + ' | ' + (f'err{7 * col}' if col < 2 else 'err0 - err7'), fontsize=10)
            ax.tick_params(labelsize=8)
            fig.colorbar(image, ax=ax, fraction=.04, pad=.03)
            images.append(image)
    heading = fig.suptitle('', fontsize=15)
    writer = imageio_ffmpeg.write_frames(str(out / filename), (1500, 1200), fps=6,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=9,
        macro_block_size=2, output_params=['-movflags', '+faststart'])
    writer.send(None)
    try:
        for it in frames:
            for row in range(4):
                for col in range(2):
                    images[row * 3 + col].set_data(R[:, :, it, col, row])
                z = D[:, :, it, row]
                alpha = np.isfinite(z).astype(float)
                if fdr:
                    alpha *= .14
                    for il in range(27):
                        dist = abs(actual[il] - time[it])
                        dist[~np.isfinite(dist)] = np.inf
                        k = np.argmin(dist)
                        if dist[k] <= .02500001:
                            alpha[il, Q[il, :, k, row] < .05] = 1
                    alpha[~np.isfinite(z)] = 0
                images[row * 3 + 2].set_data(z)
                images[row * 3 + 2].set_alpha(alpha)
            label = 'FDR corrected CFC difference' if fdr else 'Relative CFC'
            heading.set_text(f'JOJO | {label} | Time = {1000 * time[it]:.0f} ms')
            fig.canvas.draw()
            writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    finally:
        writer.close()
        plt.close(fig)
    print(f'Saved: {out / filename}', flush=True)
