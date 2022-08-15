# T0 回测图表展示系统

本sdk用于上传信号数据至T0回测视图。

## 1.使用说明

将需要上传的信号数据文件夹传入`source_files`参数，并以model_name命名。

上传后在页面上会提供 model_name - stock_id - trade_date 选项。 
```python
import t0_view_sdk
t0_view_sdk.t0_viewer.upload(model_name="test_model", source_files="./model/")
```

## 2.支持的机器

公司内网： 192.168.1.139

公司外部机器： 103.24.176.114