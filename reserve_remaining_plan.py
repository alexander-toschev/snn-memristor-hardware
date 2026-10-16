"""Pure draft routing API: no Connector imports, acquisition or executable entrypoint."""
import copy

BLOCKS=(
    dict(name='B',filter_id=2,parent_tile_id=0,taps=list(range(11)),
         positive_pool_bl=[0,1,2,4,6,10,11,13,14,15,31,3,5,7],
         negative_pool_bl=[0,9,7,5,10,1,12,4,8,31,14,2,3,6,15],
         positive_wl=4,negative_wl=5,
         preferred_positive={0:[4,7]},preferred_negative={0:[5,6],1:[5,15]},
         maximum_impacts=21421,maximum_writes=2794),
    dict(name='C',filter_id=1,parent_tile_id=0,taps=list(range(9)),
         positive_pool_bl=[0,5,6,9,11,12,14,15,31,1,2,3],
         negative_pool_bl=[2,3,5,7,8,10,11,15,9,0,1,4,12,13,31],
         positive_wl=2,negative_wl=3,
         preferred_positive={0:[2,3]},preferred_negative={0:[3,12],1:[3,13],2:[3,31]},
         maximum_impacts=18957,maximum_writes=2466),
)

PLAN=dict(protocol_id='cifar-feedback-remaining-reserves-draft-v1',draft=True,
    native_execution_enabled=False,maximum_impacts=40378,maximum_writes=5260,
    fresh_coefficients=20,imported_coefficients=280,new_partial_records=1200,
    blocks=BLOCKS,parent_bindings={'hardware':None,'simulator':None},
    parent_policy='freeze actual completed-A parent; otherwise audited baseline plus separately bound failed-A evidence',
    automatic_parent_fallback=False,maximum_sweeps_per_block=1,automatic_retry=False)


def candidate_pools(block):
    return {side+'_cell':[[block[side+'_wl'],bl] for bl in block[side+'_pool_bl']]
            for side in ('positive','negative')}


def select_route(block,quiet_cells):
    """Complete fresh screening comes first. Preferred slots reserve cells before fill."""
    pools=candidate_pools(block);n=len(block['taps']);route={};disposition=[];reservation={}
    eligible={}
    for side,pool in pools.items():
        received=[tuple(cell) for cell in quiet_cells[side]]
        if len(set(received))!=len(received) or any(list(cell) not in pool for cell in received):
            raise ValueError('Unknown or duplicate quiet-cell evidence')
        eligible[side]=[cell for cell in pool if tuple(cell) in received]
    capacity_ok=all(len(cells)>=n for cells in eligible.values())
    for short in ('positive','negative'):
        side=short+'_cell';assigned=[None]*n
        for tap,cell in block['preferred_'+short].items():
            quiet=cell in eligible[side]
            disposition.append(dict(cell=cell,tap=tap,quiet=quiet,
                status=('screened_quiet_selected_for_route' if capacity_ok else 'quiet_but_route_capacity_abort')
                       if quiet else 'screened_not_quiet',actual_cifar_assigned=False))
            if quiet:assigned[block['taps'].index(tap)]=cell
        remaining=[cell for cell in eligible[side] if cell not in assigned]
        reservation[side]=copy.deepcopy(assigned)
        if capacity_ok:
            for index,cell in enumerate(assigned):
                if cell is None:assigned[index]=remaining.pop(0)
            if len({tuple(cell) for cell in assigned})!=n:raise ValueError('Duplicate physical route')
        route[side]=assigned if capacity_ok else []
    return dict(capacity_pass=capacity_ok,route=route,preferred_disposition=disposition,
                allocation_before_fill=reservation,
                quiet_counts={side:len(cells) for side,cells in eligible.items()},
                may_issue_SET=capacity_ok)


def require_parent_bindings(plan):
    """Missing real evidence cannot become a runnable placeholder configuration."""
    if plan['draft'] or not plan['native_execution_enabled']:
        raise ValueError('Draft is not an executable acquisition protocol')
    for source,binding in plan['parent_bindings'].items():
        if not binding:raise ValueError('Missing actual parent evidence: '+source)
        if binding['source']!=source or binding['choice'] not in ('completed_A','baseline_with_failed_A'):
            raise ValueError('Invalid parent choice/source')
        evidence=[binding]
        if binding['choice']=='baseline_with_failed_A':
            if not binding.get('failed_A'):raise ValueError('Missing actual failed-A evidence')
            evidence.append(binding['failed_A'])
        for item in evidence:
            for key in ('manifest_sha256','audit_sha256'):
                digest=item.get(key,'')
                if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or len(set(digest))==1:
                    raise ValueError('Missing actual SHA: '+key)
    return copy.deepcopy(plan['parent_bindings'])
