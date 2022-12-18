## Environments

* Python 3.9.7
 
* pytorch 1.10.2

* transformers 4.17.0

* datasets 2.0.0

* RTX 3060 GPU

* CUDA 11.4

## Usage

* Download raw datasets from the above data links and put them into the directory **raw_data** like this:

	```
	--- raw_data
	  |
	  |--- medical
	  |
      |--- mrpc
      |
      |--- qqp
	```

* We have tried various pre-trained models. The following models work fine with our code:

    * model names for MRPC:
        - roberta-base
        - bert-base-uncased
        - albert-base-v2
        - microsoft/deberta-base

* Pre-process datasets.

    ```
    python ./src/preprocess.py -raw_path raw_data/mrpc
    ```
    ```
    python ./src/preprocess.py -raw_path raw_data/qqp
    ```
    ```
    python ./src/preprocess.py -raw_path raw_data/medical
    ```

## Run Our experiment
* Training and Evaluation (Baseline)

    * MRPC
    ```
    python -u src/main.py -baseline -task mrpc -model roberta-base -num_labels 2 -batch_size 16 -accum_count 1 -test_batch_size 128 >> logs/mrpc.roberta_base.baseline.log
    ```


* Training and Evaluation (DC-Match)

    * MRPC
    ```
    python -u src/main.py -task mrpc -model roberta-base -num_labels 2 -batch_size 16 -accum_count 1 -test_batch_size 128 >> logs/mrpc.roberta_base.DC_Match.log
    ```


* Training and Evaluation (DC-Match-improve)

    * MRPC
    ```
    python -u src/main.py -task mrpc -model roberta-base -num_labels 2 -batch_size 16 -accum_count 1 -test_batch_size 128 >> logs/mrpc.roberta_base.DC_Match_improve.log
    ```


* Testing (Baseline)

    * MRPC
    ```
    python -u src/main.py -baseline -task mrpc -model roberta-base -num_labels 2 -batch_size 16 -accum_count 1 -test_batch_size 128 -mode test -test_from models/mrpc/roberta-base-baseline/ -checkpoint xxx >> logs/test/mrpc.roberta_base.baseline.log
    ```

* Testing (DC-Match)

    * MRPC
    ```
    python -u src/main.py -task mrpc -model roberta-base -num_labels 2 -batch_size 16 -accum_count 1 -test_batch_size 128 -mode test -test_from models/mrpc/roberta-base-DC-Match/ -checkpoint xxx >> logs/test/mrpc.roberta_base.DC-Match.log
    ```

* Testing (DC-Match-improve)

    * MRPC
    ```
    python -u src/main.py -task mrpc -model roberta-base -num_labels 2 -batch_size 16 -accum_count 1 -test_batch_size 128 -mode test -test_from models/mrpc/roberta-base-DC-Match-improve/ -checkpoint xxx >> logs/test/mrpc.roberta_base.DC-Match-improve.log
    ```
