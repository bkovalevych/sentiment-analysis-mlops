# Sentiment Analysis on the Amazon Review Data (2018). 
## Objective 
Classify the sentiment of reviews as positive, negative by its overall mark from 0 to 5.
## Example of eval:
```
python main.py eval --text="It was very interesting. I recommend you this staff"
```
## Example of train:
```
python main.py train
```
## Configuration
Here is example of configuration. the name of file is `config.json`
```
{
  "dataset_url": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz",
  "local_path": "data/Video_Games_5.json.gz", // path to download file
  "batch_size": 5,
  "log_level": "INFO",
  "log_format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
  "train_percent": 0.7,
  "test_percent": 0.2,
  "model_path": "./model_2024-03-13_epoch_2.ptn" // path to state dict of the model. It is used for eval
}
```

## Dataset
Classification is based on Amazon Review Data (2018). You can find it by link: 
- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ .

The category of dataset is Video Games:
- https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz

## Metrics

- avg_loss: 0.22975607017676036,
- accuracy: 0.6333333333333333,
- precision: 0.4927350427350427,
- recall: 0.6333333333333333,
- f1: 0.5298801973220578