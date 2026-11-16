#!/usr/bin/env python3
"""Remove training-only pickle suffix from ONE hash-pinned official HINet checkpoint.
No unpickling/execution is performed by this utility. Subsequent load must still
use torch.load(..., weights_only=True). Storage files are copied unchanged.
"""
import argparse
import hashlib
import io
from pathlib import Path
import pickletools
import shutil
import zipfile

SOURCE_SHA='c5475d7a2a77260f8c9503b02e46a7b456bd1a7a7d7b0fc3c85ef534b473eef8'
PICKLE_SHA='e0754726b0dc2107a7542922f17c808568130ada92d2eea1fc7fd0b6dd22fa99'
CUT=404607

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def extract(source, destination):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Output must differ from source')
    if sha(source)!=SOURCE_SHA:raise ValueError('Not the supported official checkpoint hash')
    with zipfile.ZipFile(source) as zin:
        data=zin.read('archive/data.pkl')
        if hashlib.sha256(data).hexdigest()!=PICKLE_SHA:raise ValueError('Unexpected pickle')
        ops=list(pickletools.genops(data))
        current=next(i for i,(_,_,pos) in enumerate(ops) if pos==CUT)
        if not (ops[current][0].name=='BINUNICODE' and ops[current][1]=='optimizer' and ops[current-1][0].name=='BUILD'):
            raise ValueError('Unexpected boundary')
        # Original top-level prefix: EMPTY_DICT, MARK, "model", OrderedDict.
        # Model is complete at this point, so close its parent MARK with SETITEMS.
        data=data[:CUT]+b'u.'
        allowed={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch LongStorage'}
        actual={arg for op,arg,pos in pickletools.genops(data) if op.name=='GLOBAL'}
        if actual!=allowed:raise ValueError(f'Unexpected globals {actual}')
        # pickletools.dis validates stack balance, without executing pickle.
        pickletools.dis(data,out=io.StringIO())
        destination.parent.mkdir(parents=True,exist_ok=True)
        tmp=destination.with_suffix(destination.suffix+'.partial')
        try:
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as zout:
                for info in zin.infolist():
                    if info.filename=='archive/data.pkl':zout.writestr(info,data)
                    else:
                        with zin.open(info) as inp,zout.open(info,'w',force_zip64=True) as out:
                            shutil.copyfileobj(inp,out,8*1024*1024)
            tmp.replace(destination)
        finally:
            if tmp.exists():tmp.unlink()
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    print(extract(a.source,a.output));print('Load the output with weights_only=True; do not disable safe loading.')
