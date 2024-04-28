# Timestamp Join Design 

### Method 1: Go through the entire database 
Straight O(fn)
f is number of features, n is the maximum length. basically replay in time 


### Method 2: Join the tables and merge 
sample query for merging two tables: 
```
SELECT 
    test_rtx_13_camera_pose.Timestamp AS Timestamp,
    test_rtx_13_camera_pose. Value,
    test_rtx_13_arm_view. Value
FROM 
    test_rtx_13_camera_pose
LEFT JOIN 
    test_rtx_13_arm_view 
ON 
    test_rtx_13_camera_pose.Timestamp = test_rtx_13_arm_view.Timestamp

UNION

SELECT 
    test_rtx_13_arm_view.Timestamp AS Timestamp,
    NULL AS feature_1,  -- Since there's no match, feature_1 is NULL
    test_rtx_13_arm_view. Value
FROM 
    test_rtx_13_arm_view
LEFT JOIN 
    test_rtx_13_camera_pose 
ON 
    test_rtx_13_camera_pose.Timestamp = test_rtx_13_arm_view.Timestamp
WHERE 
    test_rtx_13_camera_pose.Timestamp IS NULL;
```

Note that even if we do this, this might take longer with O(fn) + O(complexity of join implementation)

### Method 3: ? 
designing a good algorithm is cool!