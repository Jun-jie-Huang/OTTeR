# OTTeR
Source Code for our EMNLP-22 findings paper "Mixed-modality Representation Learning and Pre-training for Joint Table-and-Text Retrieval in OpenQA".  We open-source a two stage OpenQA system, where it first retrieves relevant table-text blocks and then extract answers from the retrieved evidences.

## Repository Structure

- `data_ottqa`: this folder contains the original dataset copied from [OTT-QA](https://github.com/wenhuchen/OTT-QA).
- `data_wikitable`: this folder contains the crawled tables and linked passages from Wikipedia.
- `preprocessing`: this folder contains the data for training, validating and testing a code retriever. The code to obtain data for ablation study in the [paper](https://aclanthology.org/2021.acl-long.442.pdf) is also included.
- `retrieval`: this folder contains the source code for table-text retrieval stage.
- `qa`: this folder contains the source code for question answering stage
- `scripts`: this folder contains the `.py` and `.sh` files to run experiments.
- `preprocessed_data`: this folder contains the preprocessed data after preprocessing.
- `BLINK`:  this folder contains the source code adapted from `https://github.com/facebookresearch/BLINK` for entity linking.

## Requirements

```
pillow==5.4.1
torch==1.8.0
transformers==4.5.0
faiss-gpu
tensorboard==1.15.0
tqdm
torch-scatter
scikit-learn
scipy
bottle
nltk
sentencepiece
pexpect
prettytable
fuzzywuzzy
dateparser
pathos
```

We also use `apex` to support mixed precision training. You can use the following command to install `apex`.

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

## Usage

#### Step 0: Download dataset

##### Step0-1: OTT-QA dataset

```
git clone https://github.com/wenhuchen/OTT-QA.git
cp OTT-QA/release_data/* ./data_ottqa
```

##### Step0-2: OTT-QA all tables and passages

```
cd data_wikitable/
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json
cd ../
```

This script will download the crawled tables and linked passages from Wikiepdia in a cleaned format. 

### Retrieval Part -- OTTeR

#### Step 1: Preprocess

##### Step 1-1: Link table cells with passages using BLINK

We strongly suggest you to download our processed linked passage from [all_constructed_blink_tables.json](https://drive.google.com/drive/folders/1aQTOWdJ-khBm7x30y9w7LLTgT3tQ0xCy?usp=sharing) and skipping this step 1-1, since it costs too much time. You can download the `all_constructed_blink_tables.json.gz` file, then unzip it with `gunzip` and move the json file to `./data_wikitable`. After that, go tohttps://drive.google.com/drive/folders/1aQTOWdJ-khBm7x30y9w7LLTgT3tQ0xCy?usp=sharing step 2-2 to preprocess. (You can also use the linked passages [all_constructed_tables.json](https://drive.google.com/drive/folders/1aQTOWdJ-khBm7x30y9w7LLTgT3tQ0xCy?usp=sharing) following OTT-QA)

If you want to link by yourself, you can run the following script:

```
cd scripts/
for i in {0..7}
	do
	echo "Starting process", ${i}
	CUDA_VISIBLE_DEVICES=$i python link_prediction_blink.py --shard ${i}@8 --do_all --dataset ../data_wikitable/all_plain_tables.json --data_output ../data_wikitable/all_constructed_blink_tables.json 2>&1 |tee ./logs/eval$i@8.log &
done
```

Linking using the above script takes about 40-50 hours with 8 Tesla V100 32G GPUs. After linking, you can merge the 8 split files `all_constructed_blink_tables_${i}@8.json` into one json file `all_constructed_blink_tables.json`.

##### Step 1-2: Preprocess training data for retrieval

```
python retriever_preprocess.py --split train --nega intable_contra --aug_blink
python retriever_preprocess.py --split dev --nega intable_contra --aug_blink
```

These two scripts create data used for training.

##### Step 1-3: Build retrieval corpus

```
python corpus_preprocess.py
```

This script encode the whole corpus table-text blocks used for retrieval.

##### Step 1-4: Download tbid2doc file 

Download the `tfidf_augmentation_results.json.gz` file [here](https://drive.google.com/drive/folders/1aQTOWdJ-khBm7x30y9w7LLTgT3tQ0xCy?usp=sharing), then use the following command to unzip and move the unzipped json file to `./data_wikitable`. This file will be used for preprocessing in step 4-3 and step 5-1.

```
gunzip tfidf_augmentation_results.json.gz
```

#### Step2: Pretrain the OTTeR with mixed-modality synthetic pre-training

This step we pre-train the OTTeR with BART generated mixed-modality synthetic corpus. You have three choices here. 

(1) Skip Step2 and jump to Step3. In this way, you just need to remove the argument `--init_checkpoint ${PRETRAIN_MODEL_PATH}/checkpoint_best.pt` in the training script in Step3. 

(2) Download the pre-trained checkpoint (Step 2-0)

(3) Rerun the pre-training. (Step 2-1 ~ Step 2-3)

##### Step 2-0: Download pre-trained checkpoint

The pre-trained checkpoint can be found here. TODO

##### Step 2-1: Obtain table-text pairs

We use the Wikipedia hyperlinks in tables to form the table-text pairs. For simplicity, we use the crawled and preprocessed Wikipedia dumps by [OTT-QA]() TODO. You can first download through this link and then move the json file to `` 

To finetune BART with *(table-text block, real question)* , we first create finetuning data:

```
cd preprocessing/
python generate_pseudo_question.py --split train --table_pruning
python generate_pseudo_question.py --split dev --table_pruning
```

The training scripts are as follows:

```
TODO
```

Then you can create the data for BART to generate:

```
python generate_pseudo_question.py --split table_corpus_wiki --table_pruning
```

Next generate table-text pairs.

```
TODO
```

##### Step 2-2: preprocess pretraining data

```
cd preprocessing/
python prepro_bart_to_pretrain.py --split table_corpus_wiki --prefix tPrun_title --nega rand_row
python prepro_bart_to_pretrain.py --split table_corpus_wiki --nega rand_row
```

##### Step 2-3: pretraining

```
export RUN_ID=11
export BASIC_PATH=/home/t-jhuang/junjie/TableUnderstanding
export DATA_PATH=${BASIC_PATH}/preprocessed_data/pretrain
export TRAIN_DATA_PATH=${BASIC_PATH}/preprocessed_data/pretrain/bart/table_corpus_wiki_tPrun_title_rand_row.json
export DEV_DATA_PATH=${BASIC_PATH}/preprocessed_data/pretrain/dev_rand_row.json
export MODEL_PATH=${BASIC_PATH}/models/pretrain/shared_roberta_bart_rand_row
export TOKENIZERS_PARALLELISM=false
mkdir ${MODEL_PATH}
mkdir ${MODEL_PATH}/run_logs

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_pretrain_1hop_tb_retrieval.py \
  --do_train \
  --prefix ${RUN_ID} \
  --predict_batch_size 800 \
  --model_name roberta-base \
  --overwrite_cache \
  --shared_encoder \
  --no_proj \
  --train_batch_size 168 \
  --learning_rate 3e-5 \
  --fp16 \
  --train_file ${TRAIN_DATA_PATH} \
  --predict_file ${DEV_DATA_PATH}  \
  --save_tensor_path ${MODEL_PATH}/stored_training_tensors   \
  --output_dir ${MODEL_PATH} \
  --seed 1997 \
  --max_c_len 512 \
  --max_q_len 70 \
  --num_train_epochs 5 \
  --accumulate_gradients 1 \
  --gradient_accumulation_steps 1 \
  --save_checkpoints_steps 3000 \
  --eval_period 3000 \
  --warmup_ratio 0.05 \
  --num_workers 24  2>&1 |tee ${MODEL_PATH}/run_logs/retrieval_pre_training.log
```

#### Step 3: Train the OTTeR

```
export RUN_ID=0
export BASIC_PATH=.
export DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval
export TRAIN_DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval/train_intable_contra_blink_row.pkl
export DEV_DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval/dev_intable_contra_blink_row.pkl
export RT_MODEL_PATH=${BASIC_PATH}/models/otter
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/models/pretrain/shared_roberta_bart_rand_row/checkpoint-87000/
export TABLE_CORPUS=table_corpus_blink
mkdir ${RT_MODEL_PATH}

cd ./scripts
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_1hop_tb_retrieval.py \
  --do_train \
  --prefix ${RUN_ID} \
  --predict_batch_size 800 \
  --model_name roberta-base \
  --shared_encoder \
  --train_batch_size 64 \
  --fp16 \
  --init_checkpoint ${PRETRAIN_MODEL_PATH}/checkpoint_best.pt \
  --max_c_len 512 \
  --max_q_len 70 \
  --num_train_epochs 20 \
  --warmup_ratio 0.1 \
  --train_file ${TRAIN_DATA_PATH} \
  --predict_file ${DEV_DATA_PATH} \
  --output_dir ${RT_MODEL_PATH} \
  2>&1 |tee ./retrieval_training.log
```

The training step takes about 10~12 hours with 8 Tesla V100 16G GPUs.

#### Step 4: Evaluate retrieval performance

##### Step 4-1: Encode table corpus and dev. questions with OTTeR

Encode dev questions. 

```
cd ./scripts
CUDA_VISIBLE_DEVICES="0,1,2,3" python encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --shared_encoder \
    --predict_file ${BASIC_PATH}/data_ottqa/dev.json \
    --init_checkpoint ${RT_MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${RT_MODEL_PATH}/indexed_embeddings/question_dev \
    --fp16 \
    --max_c_len 512 \
    --num_workers 8  2>&1 |tee ./encode_corpus_dev.log
```

Encode table-text block corpus. It takes about 3 hours to encode.

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python encode_corpus.py \
    --do_predict \
    --encode_table \
    --shared_encoder \
    --predict_batch_size 1600 \
    --model_name roberta-base \
    --predict_file ${DATA_PATH}/${TABLE_CORPUS}.pkl \
    --init_checkpoint ${RT_MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS} \
    --fp16 \
    --max_c_len 512 \
    --num_workers 24  2>&1 |tee ./encode_corpus_table_blink.log
```

##### Step 4-2: Build index and search with FAISS

The reported results are table recalls.

```
python eval_ottqa_retrieval.py \
	 --raw_data_path ${BASIC_PATH}/data_ottqa/dev.json \
	 --eval_only_ans \
	 --query_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/question_dev.npy \
	 --corpus_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}.npy \
	 --id2doc_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}/id2doc.json \
     --output_save_path ${RT_MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json \
     --beam_size 100  2>&1 |tee ./results_retrieval_dev.log
```

##### Step 4-3: Generate retrieval output for stage-2 question answering.

This step also evaluates the table block recall defined in our paper. We use the top 15 table-text blocks for QA, i.e.,`CONCAT_TBS=15` . 

```
export CONCAT_TBS=15
python ../preprocessing/qa_preprocess.py \
     --split dev \
     --topk_tbs ${CONCAT_TBS} \
     --retrieval_results_file ${RT_MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json \
     --qa_save_path ${RT_MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
     2>&1 |tee ./preprocess_qa_dev_k100cat${CONCAT_TBS}.log;
```

### QA part -- Longformer Reader

As we mainly focus on improving retrieval accuracy in this paper, we use the state-of-the-art reader model to evaluate downstream QA performance.

#### Step 5: Train the QA model

##### Step 5-1: Create training data

As we mentioned in our paper, to balance the distribution of training data and inference data, we also takes k table-text blocks for training, which contains several ground-truth blocks and the rest of retrieved blocks. We use the following scripts to obtain the training data.

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python encode_corpus.py \
    --do_predict \
    --predict_batch_size 200 \
    --model_name roberta-base \
    --shared_encoder \
    --predict_file ${BASIC_PATH}/data_ottqa/train.json \
    --init_checkpoint ${RT_MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${RT_MODEL_PATH}/indexed_embeddings/question_train \
    --fp16 \
    --max_c_len 512 \
    --num_workers 16  2>&1 |tee ./encode_corpus_train.log

python eval_ottqa_retrieval.py \
	   --raw_data_path ${BASIC_PATH}/data_ottqa/train.json \
	   --eval_only_ans \
	   --query_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/question_train.npy \
	   --corpus_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}.npy \
	   --id2doc_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}/id2doc.json \
	   --output_save_path ${RT_MODEL_PATH}/indexed_embeddings/train_output_k100_${TABLE_CORPUS}.json \
	   --beam_size 100  2>&1 |tee ./results_retrieval_train.log

python ../preprocessing/qa_preprocess.py \
	    --split train \
	    --topk_tbs 15 \
	    --retrieval_results_file ${RT_MODEL_PATH}/indexed_embeddings/train_output_k100_${TABLE_CORPUS}.json \
	    --qa_save_path ${RT_MODEL_PATH}/train_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
	    2>&1 |tee ./preprocess_qa_train_k100.log
```

Note that we also requires the retrieval output for dev. set. You can refer to Step 4-3 to obtain the processed qa data.

##### Step 5-2: Train

```
export BASIC_PATH=.
export MODEL_NAME=mrm8488/longformer-base-4096-finetuned-squadv2
export TOPK=15
export QA_MODEL_PATH=${BASIC_PATH}/models/qa_longformer_${TOPK}_squadv2
mkdir ${QA_MODEL_PATH}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
    --do_train \
    --do_eval \
    --model_type longformer \
    --dont_save_cache \
    --overwrite_cache \
    --model_name_or_path ${QA_MODEL_PATH} \
    --evaluate_during_training \
    --data_dir ${RT_MODEL_PATH} \
    --output_dir ${QA_MODEL_PATH} \
    --train_file train_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
    --dev_file dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 4 \
    --max_seq_length 4096 \
    --doc_stride 1024 \
    --topk_tbs ${TOPK} \
    2>&1 | tee ./train_qa_longformer-base-top${TOPK}.log
```

#### Step 6: Evaluating the QA performance

```
export BASIC_PATH=.
export TOPK=15
export QA_MODEL_PATH=${BASIC_PATH}/models/qa_longformer_${TOPK}_squadv2

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
    --do_eval \
    --model_type longformer \
    --dont_save_cache \
    --overwrite_cache \
    --model_name_or_path ${MODEL_NAME} \
    --data_dir ${RT_MODEL_PATH} \
    --output_dir ${QA_MODEL_PATH} \
    --dev_file dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json\
    --per_gpu_eval_batch_size 16 \
    --max_seq_length 4096 \
    --doc_stride 1024 \
    --topk_tbs ${TOPK} \
    2>&1 | tee ./test_qa_longformer-base-top${TOPK}.log
```





### Reference

If you find our code useful to you, please cite it using the following format:

```
@article{Huang2022OTTER,
  title={Mixed-modality Representation Learning and Pre-training for Joint Table-and-Text Retrieval in OpenQA},
  author={Huang, Junjie and Zhong, Wanjun and Liu, Qian and Gong, Ming and Jiang, Daxin and Duan, Nan},
  journal={arXiv preprint arXiv:2210.05197},
  year={2022}
}
```

You can also check our another paper focusing on reasoning

```
@inproceedings{Zhong2022ReasoningOH,
  title={Reasoning over Hybrid Chain for Table-and-Text Open Domain Question Answering},
  author={Wanjun Zhong and Junjie Huang and Qian Liu and Ming Zhou and Jiahai Wang and Jian Yin and Nan Duan},
  booktitle={IJCAI},
  year={2022}
}
```



