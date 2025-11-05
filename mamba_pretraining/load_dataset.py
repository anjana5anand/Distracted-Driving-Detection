from datasets import load_dataset

dataset = load_dataset("parquet", data_files="/media/viplab/DATADRIVE1/huggingface/ucf101/test/*.parquet")


print(dataset)
