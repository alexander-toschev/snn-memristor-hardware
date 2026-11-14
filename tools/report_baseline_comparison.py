"""Offline ADC-domain comparison on saved independent MVM input combinations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Literal, cast

Row = dict[str, object]
Policy = Literal['before_only', 'bracket_centered']


def obj(value: object) -> Row:
    if not isinstance(value, dict) or not all(isinstance(k,str) for k in value):
        raise ValueError('JSON object required')
    return cast(Row,value)


def num(value: object) -> float:
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
        raise ValueError('Finite numeric value required')
    return float(value)


def residual_signal(row: Row, policy: Policy) -> float:
    before=num(obj(row['baseline_before'])['adc'])
    after=num(obj(row['baseline_after'])['adc'])
    signal=num(obj(row['signal'])['adc'])
    return signal-before if policy=='before_only' else signal-(before+after)/2


def compare(session: Path, output: Path) -> None:
    manifest=obj(json.loads((session/'manifest.json').read_text(encoding='utf-8')))
    if manifest.get('status')!='completed' or manifest.get('source')!='hardware':
        raise ValueError('Completed hardware source required')
    path=session/'observations.json'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if obj(manifest['result_sha256']).get('observations.json')!=digest:
        raise ValueError('Raw-derived observations checksum mismatch')
    values=obj(json.loads(path.read_text(encoding='utf-8')))['brackets']
    if not isinstance(values,list):
        raise ValueError('Brackets required')
    rows=[obj(v) for v in values]
    output.mkdir(parents=True,exist_ok=True)
    groups: list[Row]=[]
    summary: Row=dict(source_session=session.name,observations_sha256=digest,
                     domain='raw ADC counts',hardware_calls=0,new_programming_pulses=0,
                     fit='singleton means at delta0.025; heldout input combinations never used for fit',
                     test_labels_used=False,new_baseline_algorithm=False)
    for policy in ('before_only','bracket_centered'):
        method: Policy='before_only' if policy=='before_only' else 'bracket_centered'
        errors: list[float]=[]
        per_group: list[Row]=[]
        for context,wl in sorted({(str(r['context']),int(num(r['wl']))) for r in rows}):
            subset=[r for r in rows if r['context']==context and r['wl']==wl]
            fit: dict[int,float]={}
            for bl in (14,16,20,30,31):
                singles=[r for r in subset if r['phase']=='singleton' and r['targets']==[bl] and r['delta_requested_v']==.025]
                if len(singles)!=3:
                    raise ValueError('Three basis repeats required')
                fit[bl]=statistics.mean(residual_signal(r,method) for r in singles)
            for mask in ((16,20),(20,30,31),(14,16,20,30,31)):
                held=[r for r in subset if r['phase']=='heldout' and r['targets']==list(mask)]
                if len(held)!=3 or any(r['delta_requested_v']!=.025 for r in held):
                    raise ValueError('Three independent heldout repeats at0.025 required')
                prediction=sum(fit[b] for b in mask)
                residual=[residual_signal(r,method)-prediction for r in held]
                mean=statistics.mean(residual)
                limit=max(5.,.1*abs(prediction))
                row: Row=dict(policy=policy,context=context,wl=wl,targets=list(mask),prediction_adc=prediction,
                         residual_adc=residual,mean_residual_adc=mean,limit_adc=limit,passed=abs(mean)<=limit)
                groups.append(row);per_group.append(row);errors.extend(residual)
        summary[policy]=dict(mean_absolute_group_mean_residual_adc=statistics.mean(abs(num(r['mean_residual_adc'])) for r in per_group),
            individual_residual_rmse_adc=math.sqrt(statistics.mean(v*v for v in errors)),
            group_mean_passed=sum(r['passed'] is True for r in per_group),group_count=len(per_group),individual_count=len(errors))
    (output/'BASELINE_COMPARISON.json').write_text(json.dumps(dict(summary=summary,groups=groups),ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'BASELINE_COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    before,after=obj(summary['before_only']),obj(summary['bracket_centered'])
    text=f'''# Вычитание фона: сравнение по сохраненным измерениям

Двустороннее вычитание фона немного уменьшило ошибку на независимых комбинациях
входов. Это офлайн-пересчет одной аппаратной сессии, без новых воздействий и без
меток MNIST. Алгоритм двустороннего вычитания уже использовался в прежнем коде;
новый результат — явное сопоставление двух фиксированных способов обработки.

| Показатель | Только фон до сигнала | Фон до и после сигнала |
|---|---:|---:|
| Средний модуль средней невязки группы, АЦП |{num(before['mean_absolute_group_mean_residual_adc']):.5f}|{num(after['mean_absolute_group_mean_residual_adc']):.5f}|
| RMSE отдельных невязок, АЦП |{num(before['individual_residual_rmse_adc']):.5f}|{num(after['individual_residual_rmse_adc']):.5f}|
| Средняя невязка группы в допуске |{before['group_mean_passed']}/12|{after['group_mean_passed']}/12|

Первый способ вычисляет «сигнал минус предшествующий фон». Второй — «сигнал минус
среднее предшествующего и последующего фона». Оба используют одни и те же сырые
отсчеты АЦП. Для каждого контекста, строки и входа средний одиночный отклик
получен по трем повторам. Прогноз для комбинации равен сумме этих измеренных
одиночных откликов. Три независимых повторения каждой комбинации служат только
для проверки; они не используются при расчете прогноза.

Проверено 12 сочетаний маски и строки, всего 36 отдельных невязок. Допуск средней
невязки: максимум из 5 отсчетов АЦП и 10% модуля прогноза. Прохождение этого
условия не означает прохождение каждого повторения или пригодность группы для
MNIST. Дрейф фона остается в данных; коррекция не доказала его устранение.

Исходная проверка использовала фон 0.025 В и приращение 0.025 В. Новая расширенная
раскладка MNIST фиксирует приращение 0.05 В до теста и будет оцениваться отдельно
на собственных сохраненных откликах. Приведенные числа не являются точностью
классификации и не гарантируют улучшение MNIST.

Источник: [сессия {session.name}](https://github.com/alexander-toschev/snn-memristor-hardware/blob/codex/full-cifar-diagnostic/results/mvm_fixed_mask_diagnostic_20261022/summaries/{session.name}.json).
SHA256 observations.json: `{digest}`.
Детали: [JSON](BASELINE_COMPARISON.json), [CSV](BASELINE_COMPARISON.csv).
'''
    (output/'BASELINE_COMPARISON.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();compare(args.session,args.out)
