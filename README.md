# SMSeg

[That Slepen Al the Nyght with Open Ye! Cross-era Sequence Segmentation with Switch-memory ](https://aclanthology.org/2022.acl-long.540.pdf) at ACL2022.



## Citation

```
@inproceedings{tang-su-2022-slepen,
    title = "That Slepen Al the Nyght with Open Ye! Cross-era Sequence Segmentation with Switch-memory",
    author = "Tang, Xuemei  and
      Su, Qi",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.540",
    doi = "10.18653/v1/2022.acl-long.540",
    pages = "7830--7840",
}
```

## Requirements

* `python=3.6`
* `pytorch=1.1`

## Downloading BERT

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from. If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

Then, put the model files into ./bert-chinese

##Datasets
In this work, we use four datasets, the modern chinese datasets is MSRA, ancient chinese datastes are from http://lingcorpus.iis.sinica.edu.tw/ancient/.

We put the small sample data under the `sample_data` directory.


## Training and Testing
You can run the .sh file `run_seg.sh` to train a small model.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as encoder.
* `--classifier`:use classifier for different era datasets.
* `--switch`: use classifier is hard switch or soft switch.
* `--bert_model`: the directory of pre-trained BERT/ZEN model.
* `--use_memory`: use key-value memory networks.
* `--attention_mode`: use concat or add.
* `--decoder`: use `crf` or `softmax` as the decoder.
* `--model_name`: the name of model to save.

You can see more parameters in smseg_main.py
## Contact

Please contact us at tangxuemei@stu.pku.edu.cn if you have any questions.
Welcome to Research Center for Digital Humanities of Peking University! https://pkudh.org
# CROSSWISE-Switch-Memory-Seg
