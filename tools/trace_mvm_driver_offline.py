"""Execute ONLY extracted mode_mvm AST against inert recorders, never a board.

No private module imports, constructors or GPIO calls. The real method body is
retained; register generation is explicitly stubbed, so routing is NOT tested.
Output contains trace values/source SHA, never private source text.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np


def integers(value: object) -> list[int]:
    if not isinstance(value, (tuple,list)):
        raise ValueError('Expected byte sequence')
    out=[]
    for item in value:
        if not isinstance(item,(int,np.integer)):
            raise ValueError('Noninteger byte')
        converted=int(item)
        if not 0<=converted<=255:
            raise ValueError('Byte outside0..255')
        out.append(converted)
    return out


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str,object]]=[]

    def spi_send(self, values: object, chip: object, *other: object) -> None:
        self.events.append(dict(operation='spi_send',values=integers(values),chip=int(cast(int,chip)),other=list(other)))
        if len(self.events)>256:
            raise ValueError('Unexpected trace size')

    def spi_read(self, *arguments: object) -> list[int]:
        self.events.append(dict(operation='spi_read',arguments=list(arguments),stub_result=[0,0]))
        return [0,0]

    def ldac_mvm(self, *arguments: object) -> None:
        self.events.append(dict(operation='ldac_mvm',arguments=list(arguments)))

    def BLINK_ON(self) -> None:
        self.events.append(dict(operation='BLINK_ON',stubbed=True))

    def BLINK_OFF(self) -> None:
        self.events.append(dict(operation='BLINK_OFF',stubbed=True))

    def gen_registers_bytes(self, *arguments: object, **keywords: object) -> list[str]:
        self.events.append(dict(operation='gen_registers_bytes',routing_stubbed=True))
        return ['0b00000000','0b00000000']


@dataclass
class Context:
    fpga: Recorder
    mvm_css: list[int]


class OfflineTime:
    @staticmethod
    def sleep(seconds: object) -> None:
        return None


def extract(path: Path) -> tuple[Callable[[Context,list[int],int,int,int,int,int,int],object],list[int],str]:
    source=path.read_bytes()
    tree=ast.parse(source,filename='<private-driver-offline>')
    methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='mode_mvm']
    if len(methods)!=1:
        raise ValueError('Expected exactly one mode_mvm method')
    chips=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='mvm_css' for t in node.targets):
            value=ast.literal_eval(node.value)
            if not isinstance(value,list) or len(value)!=4 or not all(type(v) is int for v in value):
                raise ValueError('Expected literal4chip IDs')
            chips.append(value)
    if len(chips)!=1:
        raise ValueError('Expected unique literal mvm_css assignment')
    method=methods[0]
    # No import, dynamic access, file APIs or module initialization is compiled.
    allowed_calls={'int','list','range','len','min','max','abs','enumerate','zip','print','map','sum'}
    allowed_attributes={'spi_send','spi_read','ldac_mvm','BLINK_ON','BLINK_OFF','gen_registers_bytes','array','zeros','where',
                        'astype','tolist','clip','int64','int32','uint8','uint16','uint32','sleep','append'}
    for node in ast.walk(method):
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.ClassDef,ast.AsyncFunctionDef,ast.With)):
            raise ValueError('Unsafe construct in extracted method')
        if isinstance(node,ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private/dynamic attribute in extracted method')
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name) and node.func.id in allowed_calls:
                continue
            if isinstance(node.func,ast.Attribute) and node.func.attr in allowed_attributes:
                continue
            raise ValueError('Unsupported call in extracted method; inspect locally, do not bypass')
    if method.decorator_list or method.args.defaults or method.args.kw_defaults:
        raise ValueError('Unexpected decorated/defaulted method')
    method.returns=None
    for argument in method.args.args:
        argument.annotation=None
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    register_stub=Recorder()
    def quiet_print(*values: object, **keywords: object) -> None:
        return None
    builtins: dict[str,object]=dict(int=int,list=list,range=range,len=len,min=min,max=max,abs=abs,
                                 enumerate=enumerate,zip=zip,print=quiet_print,map=map,sum=sum)
    namespace: dict[str,object]={'__builtins__':builtins,'np':np,'time':OfflineTime,
        'FPGA_SIM':SimpleNamespace(gen_registers_bytes=register_stub.gen_registers_bytes)}
    exec(compile(module,'<mode_mvm-offline>','exec'),namespace)
    function=namespace['mode_mvm']
    if not callable(function):
        raise ValueError('Extracted method is not callable')
    return cast(Callable[[Context,list[int],int,int,int,int,int,int],object],function),chips[0],hashlib.sha256(source).hexdigest()


def trace(path: Path) -> dict[str,object]:
    method,chips,digest=extract(path)
    cases: list[dict[str,object]]=[]
    for label,active in [('zero',[]),*[(f'BL{b}',[b]) for b in (14,15,16,17,20,30,31)],('all32',list(range(32)))]:
        recorder=Recorder()
        context=Context(recorder,chips)
        # 0.05V at12bit/5V is41DAC. Actual private method does its own packing.
        values=[41 if b in active else 0 for b in range(32)]
        result=method(context,values,0,0,0,0,0,0)
        cases.append(dict(case=label,input_dac=values,events=recorder.events,result=result))
    return dict(schema='actual-extracted-mode-mvm-offline-trace-v1',source_sha256=digest,
        module_imported=False,private_constructor_called=False,hardware_io=False,routing_stubbed=True,
        register_generator='two binary strings 0b00000000; not a routing validation',mvm_css=chips,
        numpy_version=np.__version__,cases=cases,
        limitation='Actual method DAC packing only; no claim that returned ADC or routing matches hardware.')


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-driver-trace',action='store_true',required=True)
    parser.add_argument('--driver-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        result=trace(args.driver_file)
    except Exception as exc:
        # Do not emit traceback/source excerpts from a private file.
        result=dict(status='unsupported_offline_trace',error_type=type(exc).__name__,error=str(exc),hardware_io=False)
        code=2
    else:
        result['status']='completed'
        code=0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=result['status'],hardware_io=False)))
    return code


if __name__=='__main__':
    raise SystemExit(main())
