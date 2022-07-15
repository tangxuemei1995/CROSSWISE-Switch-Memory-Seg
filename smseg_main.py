from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from sklearn.metrics import confusion_matrix 
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from smseg_helper import get_word2id, get_gram2id, get_dicts, get_dcits_from_voc
from smseg_eval import eval_sentence, cws_evaluate_word_PRF, cws_evaluate_OOV
from smseg_model import WMSeg
import datetime
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def class_metrics(true_cls,pre_cls):
    # pre_list = []
    # for x in pre_cls:
    #     pre_list += x  
    ma = confusion_matrix(true_cls,pre_cls)
    co = 0
    for i in range(len(true_cls)):
        if true_cls[i] == pre_cls[i]:
            co += 1
    print('混淆矩阵\n:',ma)
    return co/len(true_cls)

def write_dict(name,word2id,output_model_dir):
    if not os.path.exists(output_model_dir):
        os.mkdir(output_model_dir)
    f = open(output_model_dir+'/' + name +'_voc.txt', 'w',encoding= 'utf-8')
    for key in word2id.keys():
        f.write(key + '\t' + str(word2id[key]) + '\n') 
    
    
def train(args):

    if args.use_bert and args.use_zen and args.use_lstm and args.use_trans:
        raise ValueError('We cannot use both BERT, ZEN, LSTM, TRANSFORMER')

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # if args.local_rank == -1 or args.no_cuda:
   #      device = torch.device("cpu")
   #      # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
   #      n_gpu = 0
   #      # n_gpu = torch.cuda.device_count()
   #  else:
   #      torch.cuda.set_device(args.local_rank) #local rank 进程GPU编号
   
    device = torch.device("cuda:0")
    n_gpu = 1
 

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./new_model', args.model_name + '_' + args.model_set)

    word2id = get_word2id(args.train_data_path) #the train set word -> dict，key:word value:id
    logger.info('# of word in train: %d: ' % len(word2id))
    
    dataset_map = {'MSR':0,'SGW':1,'ZGW':2,'JGW':3}
    id2dataset = {}
    for key in dataset_map.keys():
        id2dataset[dataset_map[key]] = key
 
    
    if args.use_dict:
        '''
	 use dict，only use frequency > 1 word
        '''   
        path0 = './sample_data/msr_vocab.txt'
        path1 = './sample_data/shanggu_vocab.txt'
        path2 = './sample_data/jingu_vocab.txt'
        path3 = './sample_data/zhonggu_vocab.txt'
        dicts2id, big_word2id = get_dcits_from_voc(dataset_map,path0,path1,path2,path3)              
        
        OOV_dicts2id,_ = get_dicts(args.train_data_path, args.eval_data_path,dataset_map) #from train and dev extract dicts
        logger.info('# there are : %d  dictionaries' % len(dicts2id))
        for key in dicts2id.keys():
            logger.info(key + '#  have ' + str(len(dicts2id[key]))+ ' words' )
            write_dict(key,dicts2id[key],output_model_dir)
        # exit()
    else:
        dicts2id = None
    
    # print(dicts2id['MSR'])
    # exit()
        
    
        
    # if args.use_memory:
#         '''
#         使用n-gram,只使用频次大于阈值的ngram
#         '''
#         if args.ngram_num_threshold <= 1:
#             raise Warning('The threshold of n-gram frequency is set to %d. '
#                           'No n-grams will be filtered out by frequency. '
#                           'We only filter out n-grams whose frequency is lower than that threshold!'
#                           % args.ngram_num_threshold)
#
#         gram2id = get_gram2id(args.train_data_path, args.eval_data_path,
#                               args.ngram_num_threshold, args.ngram_flag, args.av_threshold) #使用AV提取训练集和验证集/测试集ngram 然后后见ngram2id
#         logger.info('# of n-gram in memory: %d' % len(gram2id))
#     else:
#         gram2id = None


    # label_list = ['[CLS]','[SEP]']
#     begin = ['B','S','M','E']
#     pos = ['n', 'nr', 'w', 'v', 'p', 'c', 'r', 't', 'd', 'y', 'ns', 'sv', 'u', 'a', 'j', 'm', 'q', 'wv', 'f', 'yv', 's', 'mr', 'rn', 'b', 'rs', 'nn', 'nsr', 'rr']
#     for x in begin:
#         for xx in pos:
#             new =  x +'_' + xx
#             label_list.append(new)
    label_list = ["O", "B", "I", "E", "S", "[CLS]", "[SEP]"] #tag only segmentation, no pos
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    id2label = {}
    for key in label_map.keys():
        id2label[label_map[key]] = key  
        
    print(label_map,id2label)
    if args.old_model:
        print("read old model!")
        seg_model_checkpoint = torch.load('./models'+args.old_model)
        seg_model = WMSeg.from_spec(seg_model_checkpoint['spec'], seg_model_checkpoint['state_dict'], args)
    else:
        hpara = WMSeg.init_hyper_parameters(args)
        seg_model = WMSeg(word2id, big_word2id, dicts2id, label_map, dataset_map, hpara, args)

    train_examples = seg_model.load_data(args.train_data_path)
    eval_examples = seg_model.load_data(args.eval_data_path)
    num_labels = seg_model.num_labels
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input

    total_params = sum(p.numel() for p in seg_model.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    # if args.local_rank != -1:
#         num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        seg_model.half()
        
    seg_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    param_optimizer = list(seg_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    best_oov = -1
    
    history = {'epoch': [], 'class':[]}
    for key in dataset_map.keys():
        history['p' + key] = []
        history['f' + key] = []
        history['r' + key] = []
        history['oov' + key] = []
        
    num_of_no_improvement = 0
    patient = args.patient

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            seg_model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                seg_model.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue
                train_features = convert_examples_to_features(batch_examples)
                input_ids, input_mask, l_mask, label_ids, label_cls_id, matching_matrix, ngram_ids, ngram_positions, \
                segment_ids, valid_ids, word_ids, word_mask = feature2input(device, train_features)
            
                loss, _, _ = seg_model(input_ids, segment_ids, input_mask, label_ids, label_cls_id, valid_ids, l_mask, word_ids,
                                    matching_matrix, word_mask, ngram_ids, ngram_positions, device)
                    
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            seg_model.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                seg_model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true, y_cls_true = [],[]
                y_pred, y_cls_pre = [],[]
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                for start_index in range(0, len(eval_examples), args.eval_batch_size):
                    eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                         len(eval_examples))]
                    eval_features = convert_examples_to_features(eval_batch_examples)

                    input_ids, input_mask, l_mask, label_ids, label_cls_id, matching_matrix, ngram_ids, ngram_positions, \
                    segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

                    with torch.no_grad():
                        _, tag_seq, cls_pre = seg_model(input_ids, segment_ids, input_mask, labels=label_ids, labels_cls=label_cls_id,
                                                   valid_ids=valid_ids, attention_mask_label=l_mask,
                                                   word_seq=word_ids, label_value_matrix=matching_matrix,
                                                   word_mask=word_mask,
                                                   input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions, device=device)
                                    # seg_model(input_ids, segment_ids, input_mask, label_ids, label_cls_id, valid_ids, l_mask, word_ids,
                                                                                       # matching_matrix, word_mask, ngram_ids, ngram_positions,device)

                    # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
                    # logits = logits.detach().cpu().numpy()
                    logits = tag_seq.to('cpu').numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()
                    
                    labels_cls = label_cls_id.to('cpu').numpy().tolist()
                    
                    
                    
                    y_cls_true += labels_cls
                    y_cls_pre += cls_pre[0:len(labels_cls)]
                    

                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == num_labels - 1:
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])

               
                y_true_all = {}
                y_pred_all = {}
                y_pred_list = {}
                y_true_list = {}
                sentence_all = {}
                for key in dataset_map.keys():
                    y_true_all[key] = []
                    y_pred_all[key] = []
                    y_pred_list[key] = []
                    y_true_list[key] = []
                    sentence_all[key] = []
                
                
                for i in range(len(y_cls_true)):
                    index = y_cls_true[i]
                    y_true_item = y_true[i] #真实标签
                   
                    y_true_all[id2dataset[index]] += y_true_item
                    
                    # y_true_item = [id2label[i] for i in y_true_item]
                    
                    y_true_list[id2dataset[index]].append(y_true_item) 
                    
                    y_pred_item = y_pred[i] #预测标签
                    
                    y_pred_all[id2dataset[index]] += y_pred_item
                    # y_pred_item = [id2label[i] for i in y_pred_item]
                    y_pred_list[id2dataset[index]].append(y_pred_item) 
                    
                

                for example, y_true_item, y_cls_item in zip(eval_examples, y_true, y_cls_true):
                    sen = example.text_a
                    sen = sen.strip()
                    sen = sen.split(' ')
                    if len(y_true_item) != len(sen):
                        sen = sen[:len(y_true_item)]
                    sentence_all[id2dataset[y_cls_item]].append(sen)
                # print(len(y_cls_true),len(y_cls_pre))
                y_cls_pre = y_cls_pre[0:len(y_cls_true)]
                class_  = class_metrics(y_cls_true, y_cls_pre)
                ave_f, ave_p, ave_r, ave_oov = 0, 0, 0, 0 
                
                
                if not os.path.exists(output_model_dir):
                    os.mkdir(output_model_dir)
                    
                fr = open(os.path.join(output_model_dir, 'dev_metrics.tsv'), 'a', encoding='utf8')
                for key in dataset_map.keys():
                    
                    p, r, f = cws_evaluate_word_PRF(y_pred_all[key], y_true_all[key])
                    # p,r,f,oov= evaluation.get_ner_fmeasure(y_true_list[key],y_pred_list[key])
   
                    oov = cws_evaluate_OOV(y_pred_list[key], y_true_list[key], sentence_all[key], OOV_dicts2id[key]) #这里其实穿的word2id 参数不对，因为，这里是训练集和测试集所有词的都在，在这里评估的也是测试集
                    ave_f += f
                    ave_p += p
                    ave_r += r
                    ave_oov += oov
                    fr.write('\nEPoch:\t'  + str(epoch + 1) + '\t' + key + '\nbest_P:\t' + str(p) +  '\nbest_R:\t' + str(r) + '\nbest_F:\t' + str(f))
                    fr.write('\nbest_oov:\t' + str(oov) + '\n' + '\nclass:\t' + str(class_) + '\n\n')
                
                 
                
                
                    # oov = cws_evaluate_OOV(y_pred, y_true, sentence_all, word2id)
                    logger.info('OOV: %f' % oov)
                    history['epoch'].append(epoch)
                    history['p' + key].append(p)
                    history['r' + key].append(r)
                    history['f' + key].append(f)
                    history['oov' + key].append(oov)
                    history['class'].append(class_)
                    logger.info("=======entity level========")
                    logger.info("\nEpoch: %d, dataset: %d, P: %f, R: %f, F: %f, OOV: %f , class: %f", epoch + 1, dataset_map[key], p, r, f, oov, class_)
                
     
                ave_f = ave_f/len(dataset_map)
                ave_p = ave_p/len(dataset_map)
                ave_r = ave_r/len(dataset_map)
                ave_oov = ave_oov/len(dataset_map)
                
                
                logger.info("=======entity level========")
                # the evaluation method of NER
                # report = classification_report(y_true, y_pred, digits=4)

                # if args.model_name is not None:
                #     if not os.path.exists(output_model_dir):
                #         os.mkdir(output_model_dir)
                #
                #     with open(os.path.join(output_model_dir, 'dev_metrics.tsv'), 'a', encoding='utf8') as fr:
                #         fr.write('\nEPoch:\t' + str(epoch + 1) + '\nbest_P:\t' + str(p) +  '\nbest_R:\t' + str(r) + '\nbest_F:\t' + str(f))
                #         fr.write('\nbest_oov:\t' + str(oov) + '\n' + '\nclass:\t' + str(class_) + '\n')
                    #
                    # output_eval_file = os.path.join(args.model_name, "eval_results.txt")
                    #
                    # if os.path.exists(output_eval_file):
                    #     with open(output_eval_file, "a") as writer:
                    #         logger.info("***** Eval results *****")
                    #         logger.info("=======token level========")
                    #         logger.info("\n%s", report)
                    #         logger.info("=======token level========")
                    #         writer.write(report)

                if ave_f > best_f:
                    best_epoch = epoch + 1
                    best_p = ave_p
                    best_r = ave_r
                    best_f = ave_f
                    best_oov = ave_oov
                    num_of_no_improvement = 0

                    if args.model_name:
                        with open(os.path.join(output_model_dir, 'CWS_result.txt'), "w") as writer:
                            for i in range(len(y_pred)):
                                sentence = eval_examples[i].text_a
                                seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], sentence, word2id)
                                # logger.info("true: %s", seg_true_str)
                                # logger.info("pred: %s", seg_pred_str)
                                writer.write('True: %s\n' % seg_true_str)
                                writer.write('Pred: %s\n\n' % seg_pred_str)

                        best_eval_model_path = os.path.join(output_model_dir, 'model.pt')

                        if n_gpu > 1:
                            torch.save({
                                'spec': seg_model.module.spec,
                                'state_dict': seg_model.module.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                        else:
                            torch.save({
                                'spec': seg_model.spec,
                                'state_dict': seg_model.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break
        
        logger.info("\n=======best f entity level========")
        logger.info("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f\n", best_epoch, best_p, best_r, best_f, best_oov)
        logger.info("\n=======best f entity level========")
        
        with open(os.path.join(output_model_dir, 'dev_metrics.tsv'), 'a', encoding='utf8') as fw:
            fw.write('\nEPoch:\t' + str(best_epoch) + '\nbest_P:\t' + str(best_p) +  '\nbest_R:\t' + str(best_r) + '\nbest_F:\t' + str(best_f))
            fw.write('\nbest_oov:\t' + str(best_oov) + '\n')

        if os.path.exists(output_model_dir):
            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')


def test(args):


    n_gpu = 1

    device = torch.device("cuda:0")
    
    # checkpoint = torch.load(path)
 #    model.load_state_dict(checkpoint['model'])
 #    optimizer.load_state_dict(checkpoint['optimizer'])

    seg_model_checkpoint = torch.load(args.eval_model)
    # print(seg_model_checkpoint['state_dict'].keys())
#     exit()
    seg_model = WMSeg.from_spec(seg_model_checkpoint['spec'], seg_model_checkpoint['state_dict'], args)

    eval_examples = seg_model.load_data(args.test_data_path)
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input
    num_labels = seg_model.num_labels
    word2id = seg_model.word2id

    dataset_map = seg_model.dataset_map
    
    id2dataset, id2labels = {}, {}
    
    for key in dataset_map.keys():
        id2dataset[dataset_map[key]] = key
        
    
    big_word2id = seg_model.big_word2id
    dicts2id = seg_model.dict2id
    dataset_map = seg_model.dataset_map
    OOV_dicts2id,_ = get_dicts(args.train_data_path, args.eval_data_path,dataset_map) #from train and dev extract dicts
    
    
    label_map = {v: k for k, v in seg_model.labelmap.items()}
    for k in label_map.keys():
        id2labels[label_map[k]] = k
    if args.fp16:
        seg_model.half()
        
    seg_model.to(device)
    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    seg_model.to(device)

    seg_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true, y_cls_true= [], []
    y_pred, y_cls_pre = [], []

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        eval_features = convert_examples_to_features(eval_batch_examples)

        input_ids, input_mask, l_mask, label_ids, label_cls_ids, matching_matrix, ngram_ids, ngram_positions, \
        segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

        with torch.no_grad():
            _, tag_seq, cls_pre = seg_model(input_ids, segment_ids, input_mask, labels=label_ids, labels_cls=label_cls_ids,
                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                       word_seq=word_ids, label_value_matrix=matching_matrix,
                                       word_mask=word_mask,
                                       input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions, device=device)

        # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
        # logits = logits.detach().cpu().numpy()
        logits = tag_seq.to('cpu').numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        labels_cls = label_cls_ids.to('cpu').numpy().tolist()
        
        y_cls_true += labels_cls
        y_cls_pre += cls_pre[0:len(labels_cls)]
        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
                    
    y_true_all = {}
    y_pred_all = {}
    y_pred_list = {}
    y_true_list = {}
    sentence_all = {}
    for key in dataset_map.keys():
        y_true_all[key] = []
        y_pred_all[key] = []
        y_pred_list[key] = []
        y_true_list[key] = []
        sentence_all[key] = []
    

    for i in range(len(y_cls_true)):
        index = y_cls_true[i]
        y_true_item = y_true[i] #真实标签
       
        y_true_all[id2dataset[index]] += y_true_item
        
        # y_true_item = [id2label[i] for i in y_true_item]
        
        y_true_list[id2dataset[index]].append(y_true_item) 
        
        y_pred_item = y_pred[i] #预测标签
        
        y_pred_all[id2dataset[index]] += y_pred_item
        # y_pred_item = [id2label[i] for i in y_pred_item]
        y_pred_list[id2dataset[index]].append(y_pred_item) 
        
    # print(len(y_pred_list['MSR']))
    # print(len(y_pred_list['JGW']))
    # print(len(y_true_list['MSR']))
    # print(len(y_true_list['JGW']))

    for example, y_true_item, y_cls_item in zip(eval_examples, y_true, y_cls_true):
        sen = example.text_a
        sen = sen.strip()
        sen = sen.split(' ')
        if len(y_true_item) != len(sen):
            sen = sen[:len(y_true_item)]
        sentence_all[id2dataset[y_cls_item]].append(sen)
        

    
    class_  = class_metrics(y_cls_true, y_cls_pre)
    ave_f, ave_p, ave_r, ave_oov = 0, 0, 0, 0 
    
    
        
    fr = open(os.path.join(args.eval_model[0:-9], 'test_metrics.tsv'),'a+', encoding='utf8')
    
    for key in dataset_map.keys():
        
        p, r, f = cws_evaluate_word_PRF(y_pred_all[key], y_true_all[key])
        # print(len(dicts2id[key]))
        # p,r,f,oov= evaluation.get_ner_fmeasure(y_true_list[key],y_pred_list[key])
        oov = cws_evaluate_OOV(y_pred_list[key], y_true_list[key], sentence_all[key], OOV_dicts2id[key])
        ave_f += f
        ave_p += p
        ave_r += r
        ave_oov += oov
        
        print('dataset:\t',dataset_map[key] )
        print('p\t', p)
        
        print('r\t', r)
        print('f\t', f)
        print('oov\t', oov)
        print('class\t', class_)
        fr.write('\t' + key + '\nP:\t' + str(p) +  '\nR:\t' + str(r) + '\nF:\t' + str(f))
        fr.write('\noov:\t' + str(oov) + '\n' + '\nclass:\t' + str(class_) + '\n\n')
    
    

    ave_f = ave_f/len(dataset_map)
    ave_p = ave_p/len(dataset_map)
    ave_r = ave_r/len(dataset_map)
    
    ave_oov = ave_oov/len(dataset_map)
    
    with open(os.path.join(args.eval_model[0:-9], 'test_result.txt'), "w") as writer:
        for i in range(len(y_pred)):
            sentence = eval_examples[i].text_a
            seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], sentence, word2id)
            # logger.info("true: %s", seg_true_str)
            # logger.info("pred: %s", seg_pred_str)
            writer.write(id2dataset[y_cls_true[i]]+'\tTrue:\t' + seg_true_str +'\n')
            writer.write(id2dataset[y_cls_true[i]]+'\tPred:\t' + seg_pred_str + '\n')
    

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--eval_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--old_model",
                        default=None,
                        type=str,
                        help="training based on lod model")
    parser.add_argument("--use_lstm",
                        action='store_true',
                        help="Whether to use bi-lstm.")
			   
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
			   
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
			   
    parser.add_argument("--use_trans",
                        action='store_true',
                        help="Whether to use transformer.")
			   
    parser.add_argument("--bert_model", default='bert-base-chinese', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument('--ngram_num_threshold', type=int, default=0, help="The threshold of n-gram frequency")
    parser.add_argument('--av_threshold', type=int, default=5, help="av threshold")
    parser.add_argument('--max_ngram_length', type=int, default=5,
                        help="The maximum length of n-grams to be considered.")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument('--model_set', type=str, default=None, help="")
    parser.add_argument("--use_memory",action='store_true',help="Whether to run training.")
    parser.add_argument("--use_dict",action='store_true', help="use dictionary or not")
    parser.add_argument("--switch",type=str, default='', help="use classifier is hard switch or soft switch")
    parser.add_argument("--classifier",action='store_true', help="use classifier for different era dataset or not")
    parser.add_argument('--decoder', type=str, default='softmax',
                        help="the decoder to be used, should be one of softmax and crf.")
    parser.add_argument('--ngram_flag', type=str, default='av,ngram tool', help="")
    parser.add_argument('--attention_mode', type=str, default='add', help="use concat or add")
    parser.add_argument('--save_top',
                        type=int,
                        default=1,
                        help="")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    elif args.do_predict:
        predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
