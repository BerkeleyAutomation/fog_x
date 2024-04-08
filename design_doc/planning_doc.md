# TODOs 

### Small Steps 
2. storage implementation
4. loading from the existing stored data
5. efficient image storage 
6. compare with standard tfds on loading and storage

### known bugs 
1. when creating a new dataset, the first entry is null 
2. merging and loading from df lost the schema and original data types 
3. sql part is completely broken 

### Big Steps 
1. asynchonous writing, test with multiple processes & episodes 
2. querying language and pulling optimization
3. time sync join policy 
4. distributed storage (rocksdb?)
5. handling streaming with a builder pattern (?)
6. lazy loading
7. ROS / ROS2 