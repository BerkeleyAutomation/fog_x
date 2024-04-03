# Table Design Spec of SQL Database 

## Per Dataset 
### Metadata 
Stored as a separate json file with field: owner, license, etc. 

### Metadata Table 

```
Table Name: Metadata 
Column: Episode_ID | Episode_Description | # TODO: start_time, duration, other user defined metadata
```


## Per Episode

### Feature Table(s)
```
Table Name: `Episode_ID`_`Feature_Name`
Column: Timestamp | Storage_ID(String) | Storage_ID(String)*
```
Here we assume the features may come as tuples (such as dataset collected on mulitple robots at the same time ), so we can group (join) them together.

### (Post Processing) Dataset
```
Table Name: `Episode_ID`_Dataset
Column: Timestamp | Feature_1 | Feature_2 | ...
```