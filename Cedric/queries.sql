-- SQL querries 

-- T0 export overall
select (case when trunc(updated_at-created_at)<=5 then trunc(updated_at-created_at) else 6 end), sum(nvl(case when b.name='T0 RAW to T1 tape' then 1 end, 0)) as "RAW",
sum(nvl(case when b.name like 'T0 AOD to T%' then 1 end, 0)) as "AOD",
sum(nvl(case when b.name like 'T0 ESD %' then 1 end, 0)) as "ESD",
sum(nvl(case when b.name like 'T0 DESD to T%' then 1 end, 0)) as "DESD",
sum(nvl(case when b.name like 'T0 DAOD to T%' then 1 end, 0)) as "DAOD",
sum(nvl(case when b.name like 'T0 DRAW to T%' then 1 end, 0)) as "DRAW"
from atlas_rucio.locks a,
(select r.id, s.name from atlas_rucio.subscriptions s, atlas_rucio.rules r where s.name like 'T0%' and s.state!='I'
and r.subscription_id=s.id and r.created_at>to_date('01-07-2016', 'DD-MM-YYYY') and r.created_at<to_date('01-08-2016', 'DD-MM-YYYY')
and r.activity='T0 Export') b
where a.rule_id=b.id
group by (case when trunc(updated_at-created_at)<=5 then trunc(updated_at-created_at) else 6 end)
order by (case when trunc(updated_at-created_at)<=5 then trunc(updated_at-created_at) else 6 end)

-- T0 to Castor
select trunc((updated_at-created_at)*24), sum(nvl(case when b.name like '%.RAW%' then 1 end, 0)) as "RAW",
sum(nvl(case when b.name like '%.AOD%' then 1 end, 0)) as "AOD",
sum(nvl(case when b.name like '%.ESD%' then 1 end, 0)) as "ESD",
sum(nvl(case when b.name like '%.DESD%' then 1 end, 0)) as "DESD",
sum(nvl(case when b.name like '%.DAOD%' then 1 end, 0)) as "DAOD",
sum(nvl(case when b.name like '%.DRAW%' then 1 end, 0)) as "DRAW"
from atlas_rucio.locks a,
(select r.id, r.name from atlas_rucio.rules r where r.created_at>to_date('01-07-2016', 'DD-MM-YYYY') and r.created_at<to_date('01-08-2016', 'DD-MM-YYYY')
and activity='T0 Tape') b
where a.rule_id=b.id 
group by trunc((updated_at-created_at)*24)
order by trunc((updated_at-created_at)*24)

