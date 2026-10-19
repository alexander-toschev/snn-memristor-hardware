"""Lossless frequent durable raw checkpoints; partial indexes never imply stage completion."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil


def simulated_process_loss(writer,source,stage,positions,impacts):
    """Explicit qualification fault: SIM only; never a hardware recovery operation."""
    if source!='simulator' or stage!=1 or positions!=17:
        raise ValueError('Process-loss injection is fixed to SIM stage1 after17positions')
    atomic_json(writer.directory.parent/'fault_injection.json',dict(schema='cifar-sim-process-loss-v1',
        source=source,stage=stage,software_observed_positions=positions,software_observed_impacts=impacts,
        raw_index_sha256=sha(writer.directory/'index.json'),exit_code=86,
        interpretation='Marker is a software observation; only the closed-index raw is durable evidence. No close or final manifest is claimed.'))
    os._exit(86)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    if os.name!='nt':
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)


class Writer:
    def __init__(self,directory,limit=8*1024**2,maximum_expanded_bytes=2*1024**3,minimum_free_bytes=8*1024**3):
        if not 1<=limit<=64*1024**2:raise ValueError('Invalid lossless chunk limit')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.limit=limit;self.maximum=maximum_expanded_bytes;self.reserve=minimum_free_bytes
        self.parts=[];self.stream=None;self.total=0;self.expanded=0;self.all_hash=hashlib.sha256();self.finished=False
        self.checkpoint(reason='writer_created')

    def _close(self):
        if self.stream is None:return
        self.stream.close();self.file.flush();os.fsync(self.file.fileno());self.file.close()
        path=self.directory/self.name
        if path.stat().st_size>128*1024**2:raise ValueError('Compressed raw chunk too large')
        self.parts.append(dict(path=self.name,compressed_bytes=path.stat().st_size,compressed_sha256=sha(path),
                               bytes=self.size,sha256=self.part_hash.hexdigest(),records=self.records))
        self.stream=None

    def checkpoint(self,**progress):
        self._close()
        index=dict(schema='lossless-chunked-jsonl-v1',parts=self.parts,records=self.total,sha256=self.all_hash.hexdigest(),
                   gzip_compression_level=1,stream_finalized=self.finished,
                   checkpoint_semantics='Durable closed parts only; not an acquisition-completion or within-stage-resume claim')
        atomic_json(self.directory/'index.json',index)
        atomic_json(self.directory.parent/'progress.json',dict(records=self.total,expanded_bytes=self.expanded,
                    closed_parts=len(self.parts),raw_index_sha256=sha(self.directory/'index.json'),**progress))
        return index

    def write(self,record):
        if self.finished:raise ValueError('Raw writer already finalized')
        line=(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if len(line)>self.limit or self.expanded+len(line)>self.maximum:raise ValueError('Finite raw storage budget exhausted')
        if self.total%4096==0 and shutil.disk_usage(self.directory).free<self.reserve:raise ValueError('Raw disk reserve reached')
        if self.stream and self.size+len(line)>self.limit:self.checkpoint(reason='raw_chunk_boundary')
        if self.stream is None:
            self.name='part-%06d.jsonl.gz'%len(self.parts);self.file=(self.directory/self.name).open('xb')
            self.stream=gzip.GzipFile(filename='',mode='wb',fileobj=self.file,mtime=0,compresslevel=1)
            self.size=self.records=0;self.part_hash=hashlib.sha256()
        self.stream.write(line);self.size+=len(line);self.records+=1;self.total+=1;self.expanded+=len(line)
        self.part_hash.update(line);self.all_hash.update(line)

    def finish(self):
        self.finished=True
        return self.checkpoint(reason='stream_finalized_not_stage_qualification')
