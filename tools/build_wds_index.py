"""WebDataset索引生成器(CPU premium, blobfuse串行扫, gzip缓存).
blobfuse扫慢(~113min)但CPU不被抢+不浪费GPU. 一次性, gzip缓存后训练秒载.
每tar只读首个json拿ch_names(同tar全同, 实测) + getnames()拿key.
"""
import argparse, gzip, json, os, pickle, tarfile, time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--mode', default='train')
    ap.add_argument('--max_patches', type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    index = []
    shape_groups = defaultdict(list)
    n_tar = errs = 0
    buckets = sorted(b for b in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, b)))
    print(f"[index] {len(buckets)} buckets, 开始串行扫", flush=True)
    for bucket in buckets:
        bdir = os.path.join(args.root, bucket)
        for fn in sorted(os.listdir(bdir)):
            if not fn.endswith('.tar'):
                continue
            tar_path = os.path.join(bdir, fn)
            try:
                tf = tarfile.open(tar_path)
                members = tf.getmembers()
                jm = [m for m in members if m.name.endswith('.json')]
                if jm:
                    meta0 = json.load(tf.extractfile(jm[0]))
                    n_ch = meta0['n_channels']
                    n_p = min(meta0['n_patches'], args.max_patches)
                    raw_ch = meta0['ch_names']
                    ch_names = tuple(c.upper() for c in raw_ch)
                    sg_key = (n_ch, n_p, ch_names)
                    for m in jm:
                        key = m.name[:-5]
                        i = len(index)
                        index.append((tar_path, key, n_ch, n_p, raw_ch))
                        shape_groups[sg_key].append(i)
                tf.close()
            except Exception as e:
                errs += 1
                if errs <= 20:
                    print(f"  [warn] {tar_path}: {e}", flush=True)
            n_tar += 1
            if n_tar % 200 == 0:
                print(f"[index] {n_tar} tars, {len(index)} samples, {time.time()-t0:.0f}s", flush=True)

    print(f"[index] 完成: {len(index)} samples, {len(shape_groups)} groups, {errs} errs, {time.time()-t0:.1f}s", flush=True)

    idx_cache = os.path.join(args.root, f".wds_index_{args.mode}.pkl.gz")
    tmp = idx_cache + f".tmp{os.getpid()}"
    t1 = time.time()
    with gzip.open(tmp, 'wb', compresslevel=4) as fh:
        pickle.dump({'index': index, 'shape_groups': dict(shape_groups)}, fh)
    os.replace(tmp, idx_cache)
    sz = os.path.getsize(idx_cache) / 1e6
    print(f"[index] 缓存 {idx_cache} ({sz:.1f}MB, 写{time.time()-t1:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
