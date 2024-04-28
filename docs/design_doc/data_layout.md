# Data Layout 

Consisting of an episode table and a step table. Step table is partitioned. 

Episode table contains information that is consistent across the step, such as 
feature types, tags. 

Step table contains the actual data. 