"""Small bounded pilot trace; retain invalid native responses without pickle."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil


class Writer:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        if shutil.disk_usage(self.directory).free<2*1024**3:raise OSError('Need2GiB raw reserve')
        self.file=(self.directory/'raw.jsonl.gz').open('xb')
        self.stream=gzip.GzipFile(fileobj=self.file,mode='wb',compresslevel=1,mtime=0)
        self.hash=hashlib.sha256();self.size=self.lines=0;self.finished=False
    def write(self,row):
        data=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if self.size+len(data)>64*1024**2:raise OSError('64MiB expanded raw cap')
        if self.lines%256==0 and shutil.disk_usage(self.directory).free<1024**3:raise OSError('1GiB free reserve')
        self.stream.write(data);self.hash.update(data);self.size+=len(data);self.lines+=1
    def finish(self):
        if self.finished:return
        self.stream.close();self.file.close();self.finished=True
        path=self.directory/'raw.jsonl.gz'
        value=dict(lines=self.lines,expanded_bytes=self.size,expanded_sha256=self.hash.hexdigest(),
                   compressed_bytes=path.stat().st_size,compressed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),compression='gzip-level1')
        with (self.directory/'index.json').open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
