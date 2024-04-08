# System Assumptions 
Fox manages a trajectory information table that contains summary, tagging, etc and a step data table that contains all the data (images, etc).  
1. Episode information metadata should fit in memory. 
2. Trajectory data can go beyond memory or hardware disks. 
3. All trajectory data within an episode should fit in memory (TODO: this constraint should be relaxed)

### Consistency
1. Data can be collected distributedly on multiple robots /processes. 
2. No consistency is guaranteed if the data is changed after calling close() 
3. No consistency is guanrateed if reading and writing to the same data at the same time 