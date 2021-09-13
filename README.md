# OTTeR
Source Code for paper "Joint Table-Text Retrieval for Open-domain Question Answering".  We open-source a two stage OpenQA system, where it first retrieves relevant table-text blocks and then extract answers from the retrieved evidences.

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

We strongly suggest that you downloading the linked dataset from this link and skipping this step, since it costs too much time. You can download the files from this link TODO.

The running script:

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
python retriever_preprocess.py --split train --nega intable_contra --replace_link_passages --aug_blink
python retriever_preprocess.py --split dev --nega intable_contra --replace_link_passages --aug_blink
```

These two scripts create data used for training.

##### Step 1-3: Build retrieval corpus

```
python corpus_preprocess.py --split table_corpus_blink
```

This script creates corpus data used for inference.

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
export MODEL_PATH=${BASIC_PATH}/models/otter
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/models/pretrain/shared_roberta_bart_rand_row/checkpoint-87000/
export TOKENIZERS_PARALLELISM=false
export TABLE_CORPUS=table_corpus_blink
mkdir ${MODEL_PATH}
mkdir ${MODEL_PATH}/run_logs
mkdir ${MODEL_PATH}/run_logs/${TABLE_CORPUS}

cd script/
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_1hop_tb_retrieval.py \
  --do_train \
  --prefix ${RUN_ID} \
  --predict_batch_size 800 \
  --model_name roberta-base \
  --shared_encoder \
  --no_proj \
  --normalize_table \
  --three_cat \
  --part_pooling first \
  --one_query \
  --train_batch_size 64 \
  --learning_rate 2e-5 \
  --fp16 \
  --train_file ${TRAIN_DATA_PATH} \
  --predict_file ${DEV_DATA_PATH}  \
  --init_checkpoint ${PRETRAIN_MODEL_PATH}/checkpoint_best.pt \
  --save_tensor_path ${BASIC_PATH}/models/continue_train/shared_roberta_bart_threecat_basic_mean_one_query_row_corpus_wiki_3000_newTensors/stored_training_tensors   \
  --output_dir ${MODEL_PATH} \
  --seed 1997 \
  --eval_period -1 \
  --max_c_len 512 \
  --max_q_len 70 \
  --metadata \
  --psg_mode ori \
  --num_train_epochs 20 \
  --accumulate_gradients 1 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.1 \
  --num_workers 24  2>&1 |tee ${MODEL_PATH}/run_logs/retrieval_training.log
```

The training step takes about 10~12 hours with 8 Tesla V100 16G GPUs.

#### Step 4: Evaluate retrieval performance

##### Step 4-1: Encode table corpus and dev. questions with OTTeR

Encode dev questions. 

```
CUDA_VISIBLE_DEVICES="0,1,2,3" python encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --shared_encoder \
    --no_proj \
    --metadata \
    --normalize_table \
    --three_cat \
    --part_pooling first \
    --one_query \
    --predict_file ${BASIC_PATH}/data_ottqa/dev.json \
    --init_checkpoint ${MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${MODEL_PATH}/indexed_embeddings/question_dev \
    --fp16 \
    --max_c_len 512 \
    --num_workers 8  2>&1 |tee ${MODEL_PATH}/run_logs/encode_corpus_dev.log
```

Encode table-text block corpus. It takes about 3 hours to encode.

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python encode_corpus.py \
    --do_predict \
    --encode_table \
    --metadata \
    --psg_mode ori \
    --shared_encoder \
    --no_proj \
    --normalize_table \
    --three_cat \
    --part_pooling first \
    --one_query \
    --predict_batch_size 1600 \
    --model_name roberta-base \
    --predict_file ${DATA_PATH}/${TABLE_CORPUS}.pkl \
    --init_checkpoint ${MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS} \
    --fp16 \
    --max_c_len 512 \
    --num_workers 24  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/encode_corpus_table_blink.log
```

##### Step 4-2: Build index and search with FAISS

The reported results are table recalls.

```
python eval_ottqa_retrieval.py \
	 --raw_data_path ${BASIC_PATH}/data_ottqa/dev.json \
	 --eval_only_ans \
	 --three_cat \
	 --query_embeddings_path ${MODEL_PATH}/indexed_embeddings/question_dev.npy \
	 --corpus_embeddings_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}.npy \
	 --id2doc_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}/id2doc.json \
     --output_save_path ${MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json \
     --beam_size 100  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/results_retrieval_dev.log
```

##### Step 4-3: Generate retrieval output for stage-2 question answering.

This step also evaluates the table block recall defined in our paper. We use the top 15 table-text blocks for QA, i.e.,`CONCAT_TBS=15` .

```
for CONCAT_TBS in 1 5 10 15 20 30 50 100;
do
python ../preprocessing/qa_preprocess.py \
     --split dev \
  --reprocess \
     --topk_tbs ${CONCAT_TBS} \
     --retrieval_results_file ${MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json \
     --qa_save_path ${MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
     2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_dev_k100cat${CONCAT_TBS}.log;
done
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
    --no_proj \
    --metadata \
    --normalize_table \
    --three_cat \
    --part_pooling first \
    --one_query \
    --predict_file ${BASIC_PATH}/data_ottqa/train.json \
    --init_checkpoint ${MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${MODEL_PATH}/indexed_embeddings/question_train \
    --fp16 \
    --max_c_len 512 \
    --num_workers 16  2>&1 |tee ${MODEL_PATH}/run_logs/encode_corpus_train.log

python eval_ottqa_retrieval.py \
	   --raw_data_path ${BASIC_PATH}/data_ottqa/train.json \
	   --eval_only_ans \
	   --three_cat \
	   --query_embeddings_path ${MODEL_PATH}/indexed_embeddings/question_train.npy \
	   --corpus_embeddings_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}.npy \
	   --id2doc_path ${MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}/id2doc.json \
	   --output_save_path ${MODEL_PATH}/indexed_embeddings/train_output_k100_${TABLE_CORPUS}.json \
	   --beam_size 100  2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/results_retrieval_train.log
python ../preprocessing/qa_preprocess.py \
	    --split train \
	    --topk_tbs 15 \
	    --retrieval_results_file ${MODEL_PATH}/indexed_embeddings/train_output_k100_${TABLE_CORPUS}.json \
	    --qa_save_path ${MODEL_PATH}/train_preprocessed_${TABLE_CORPUS}_k100.json \
	    2>&1 |tee ${MODEL_PATH}/run_logs/${TABLE_CORPUS}/preprocess_qa_train_k100.log
```

Note that we also requires the retrieval output for dev. set. You can refer to Step 4-3 to obtain the processed qa data.

##### Step 5-2: Train

```
export RUN_ID=2
export BASIC_PATH=.
export MODEL_NAME=mrm8488/longformer-base-4096-finetuned-squadv2
export TOKENIZERS_PARALLELISM=false
export TOPK=15
export MODEL_PATH=${BASIC_PATH}/models/qa_model/longformer_pretrain_rand_row_87000_first_blink_${TOPK}_squadv2
mkdir ${MODEL_PATH}
mkdir ${MODEL_PATH}/run_logs

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
    --do_train \
    --do_eval \
    --model_type longformer \
    --dont_save_cache \
    --repreprocess \
    --overwrite_cache \
    --model_name_or_path ${MODEL_NAME} \
	--add_special_tokens \
    --evaluate_during_training \
    --data_dir ${BASIC_PATH}/models/continue_train/shared_roberta_bart_row_corpus_wiki_87000_threecat_basic_one_query_first \
    --output_dir ${MODEL_PATH} \
    --train_file train_preprocessed_table_corpus_blink_k100cat15.json \
    --dev_file dev_preprocessed_table_corpus_blink_k100cat15.json \
    --prefix pretrain_row_87000 \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 4 \
    --max_seq_length 4096 \
    --doc_stride 1024 \
    --num_tokenizer_vocab 50272 \
    --topk_tbs ${TOPK} \
    --threads 24  2>&1 | tee ${MODEL_PATH}/run_logs/train_qa_longformer-base-pretrain87000-blink-top${TOPK}.log
```

#### Step 6: Evaluating the QA performance

```
export BASIC_PATH=.
export TOKENIZERS_PARALLELISM=false
export TOPK=15
export MODEL_PATH=${BASIC_PATH}/models/qa_model/longformer_pretrain_rand_row_87000_first_blink_${TOPK}_squadv2
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_final_qa.py \
    --do_eval \
    --model_type longformer \
    --dont_save_cache \
    --model_name_or_path ${MODEL_NAME} \
	--add_special_tokens \
    --evaluate_during_training \
    --data_dir ${BASIC_PATH}/models/continue_train/shared_roberta_bart_row_corpus_wiki_87000_threecat_basic_one_query_first \
    --output_dir ${MODEL_PATH} \
    --train_file train_preprocessed_table_corpus_blink_k100cat15.json \
    --dev_file dev_preprocessed_table_corpus_blink_k100cat15.json\
    --prefix pretrain_row_87000 \
    --per_gpu_eval_batch_size 16 \
    --max_seq_length 4096 \
    --doc_stride 1024 \
    --num_tokenizer_vocab 50272 \
    --topk_tbs ${TOPK} \
    --repreprocess \
    --overwrite_cache \
    --threads 24  2>&1 | tee ${MODEL_PATH}/run_logs/test_qa_longformer-base-pretrain87000-blink-top${TOPK}.log
```





### Reference

If you find this project useful, please cite it using the following format

```

```

