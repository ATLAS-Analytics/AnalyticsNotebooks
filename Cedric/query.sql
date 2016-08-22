-- Query 1 
select count(*), sum(nvl(mlength, 0)), sum(nvl(mbytes, 0)), atlas_rucio.id2rse(rse_id) from (select b.scope, b.name, b.rse_id, max(length) as mlength, max(bytes) as mbytes from atlas_rucio.rules a, atlas_rucio.dataset_locks b
where eol_at<sysdate and a.id=b.rule_id group by b.scope, b.name, b.rse_id) group by rse_id order by sum(nvl(mbytes, 0)) desc

-- Query 2
select count(*), sum(mlength), sum(mbytes), trunc(meol_at, 'MM') from (select b.scope, b.name, b.rse_id,  nvl(max(length), 0) as mlength, nvl(max(bytes), 0) as mbytes, max(eol_at) as meol_at
from atlas_rucio.rules a, atlas_rucio.dataset_locks b
where eol_at is not null and a.id=b.rule_id group by b.scope, b.name, b.rse_id) group by trunc(meol_at, 'MM') order by trunc(meol_at, 'MM')

-- Query 3
select count(*), sum(mlength), sum(mbytes), atlas_rucio.id2rse(rse_id), scope from (select b.scope, b.name, b.rse_id, nvl(max(length), 0) as mlength, nvl(max(bytes), 0) as mbytes from atlas_rucio.rules a, atlas_rucio.dataset_locks b
where eol_at<sysdate and a.id=b.rule_id group by b.scope, b.name, b.rse_id) group by rse_id, scope

-- Query 4
select count(*), sum(mlength), sum(mbytes), scope from (select b.scope, b.name, b.rse_id, nvl(max(length), 0) as mlength, nvl(max(bytes), 0) as mbytes from atlas_rucio.rules a, atlas_rucio.dataset_locks b
where eol_at<sysdate and a.id=b.rule_id group by b.scope, b.name, b.rse_id) group by scope

-- Query 5
select count(*), sum(mlength), sum(mbytes), scope, trunc(meol_at, 'MM') from (select b.scope, b.name, b.rse_id, nvl(max(length), 0) as mlength, 
nvl(max(bytes), 0) as mbytes, max(eol_at) as meol_at
from atlas_rucio.rules a, atlas_rucio.dataset_locks b
where eol_at<sysdate and a.id=b.rule_id and eol_at is not null group by b.scope, b.name, b.rse_id) group by trunc(meol_at, 'MM'), scope