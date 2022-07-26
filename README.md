# Noval-character-extraction

基于`Amazon SageMaker`进行 `小说对话定位` 的任务 和 `小说角色提取`的任务，并且利用`Amazon SageMaker pipeline` 进行工程化部署， 支持实时推理和批量推理

## Data
数据为私有数据， 无法公开

## Model
* bert
* t5
* longformer [模型论文](https://arxiv.org/abs/2004.05150)

## code structure
```
source--|--modela
        |--modelb
        |--longformer
        |--mlops
```