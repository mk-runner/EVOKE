import os
from modules.tokenizers_new import build_my_tokenizer
from modules.dataloaders_v0401 import (FinetuneLoaderHasIndication, FinetuneLoaderNotIndication, PretrainLoader)
from modules.metrics.metrics import compute_all_scores
from modules.optimizers import build_two_stage_optimizer, build_lr_scheduler
from modules.trainer_v0401 import PTrainer, FTrainer, Tester
from modules.utils import setup_arguments, setup_seed
from models.model_pretrain_finetune_v0425_ablation import FineTune, Pretrain

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    args, logger = setup_arguments()
    # -------------------------------
    # fix random seeds
    # -------------------------------
    setup_seed(args["seed"])
    # -------------------------------
    logger.info('start load data...')
    # -------------------------------
    # create tokenizer
    # -------------------------------
    print("load tokenizer...")
    tokenizer = build_my_tokenizer(tokenizer_dir=args['tokenizer_dir'], data_name=args['data_name'],
                                   ann_path=args['ann_path'], is_same_tokenizer=True)
    args['vocab_size'] = tokenizer.get_vocab_size()
    args['suppress_UNK'] = tokenizer.token_to_id('[UNK]')  # used for the CMN or r2gen text decoder
    # -------------------------------
    # save the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    logger.info(params)
    print(params)
    # -------------------------------
    # create data loader
    # -------------------------------
    if args['task'] == 'pretrain':
        train_loader = PretrainLoader(args, tokenizer, split='train', shuffle=True)
        val_loader = PretrainLoader(args, tokenizer, split='val', shuffle=False, drop_last=False)
        test_loader = PretrainLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)
    elif args['task'] == 'finetune':
        train_loader_inc, val_loader_inc, test_loader_inc = None, None, None
        if args['is_add_indication']:
            train_loader_inc = FinetuneLoaderHasIndication(args, tokenizer, split='train', shuffle=True)
            val_loader_inc = FinetuneLoaderHasIndication(args, tokenizer, split='val', shuffle=False, drop_last=False)
            test_loader_inc = FinetuneLoaderHasIndication(args, tokenizer, split='test', shuffle=False, drop_last=False)

        # has similar historical cases and not has indication
        train_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='train', shuffle=True)
        val_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='val', shuffle=False, drop_last=False)
        test_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='test', shuffle=False, drop_last=False)
    else:  # test
        train_loader_inc, train_loader_not_inc = None, None
        val_loader_inc, val_loader_not_inc = None, None
        test_loader_inc = None
        if args['is_add_indication']:
            test_loader_inc = FinetuneLoaderHasIndication(args, tokenizer, split='test', shuffle=False, drop_last=False)
        test_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='test', shuffle=False, drop_last=False)
    if args['task'] in ['pretrain', 'pretrain_inference']:
        num_train = len(train_loader.dataset) if train_loader is not None else 'None'
        num_val = len(val_loader.dataset) if val_loader is not None else 'None'
        num_test = len(test_loader.dataset) if test_loader is not None else 'None'
        print(f"the number of train_data: {num_train}, "
              f"valid_data: {num_val}, "
              f"test_data: {num_test}, ")
        logger.info(f"the number of train_data: {num_train}, "
                    f"valid_data: {num_val}, "
                    f"test_data: {num_test}, ")
    else:
        num_train_inc = len(train_loader_inc.dataset) if train_loader_inc is not None else 'None'
        num_train_not_inc = len(train_loader_not_inc.dataset) if train_loader_not_inc is not None else 'None'
        num_val_inc = len(val_loader_inc.dataset) if val_loader_inc is not None else 'None'
        num_val_not_inc = len(val_loader_not_inc.dataset) if val_loader_not_inc is not None else 'None'
        num_test_inc = len(test_loader_inc.dataset) if test_loader_inc is not None else 'None'
        num_test_not_inc = len(test_loader_not_inc.dataset) if test_loader_not_inc is not None else 'None'
        print(f"the number of train_data multiview (indication-not_indication): {num_train_inc}-{num_train_not_inc}, "
              f"valid_data multiview (indication-not_indication): {num_val_inc}-{num_val_not_inc}, "
              f"test_data multiview (indication-not_indication): {num_test_inc}-{num_test_not_inc}, ")
        logger.info(f"the number of train_data (indication-not_indication): {num_train_inc}-{num_train_not_inc}, "
                    f"valid_data (indication-not_indication): {num_val_inc}-{num_val_not_inc}, "
                    f"test_data (indication-not_indication): {num_test_inc}-{num_test_not_inc}, ")

    # -------------------------------
    # build model architecture
    # -------------------------------
    if args['task'] == 'pretrain':
        model = Pretrain(args, tokenizer, args['data_name'])
    else:  # finetune or test
        model = FineTune(args, tokenizer, args['data_name'])
    model = model.to(args['device'])
    # -------------------------------
    print(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    logger.info(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    # get function handles of loss and metrics
    # -------------------------------
    metrics = compute_all_scores
    # -------------------------------
    # build optimizer, learning rate scheduler
    # -------------------------------
    optimizer = build_two_stage_optimizer(args, model)
    # optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # -------------------------------
    # build trainer and start to train
    logger.info(f'start {args["task"]}!')
    print(f'start {args["task"]}!')
    # -------------------------------
    if args['task'] == 'pretrain':
        kwarg = {"model": model, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
                 "lr_scheduler": lr_scheduler, "train_loader": train_loader,
                 "val_loader": val_loader, "test_loader": test_loader,
                 "logger": logger, "task": args['task'],
                 'is_save_checkpoint': args['is_save_checkpoint']}
    else:
        kwarg = {"model": model, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
                 "lr_scheduler": lr_scheduler, "train_loader_inc": train_loader_inc,
                 "train_loader_not_inc": train_loader_not_inc, "val_loader_inc": val_loader_inc,
                 "val_loader_not_inc": val_loader_not_inc, "test_loader_inc": test_loader_inc,
                 "test_loader_not_inc": test_loader_not_inc, "logger": logger, "task": args['task'],
                 'is_save_checkpoint': args['is_save_checkpoint']}

    if args['task'] == 'pretrain':
        trainer = PTrainer(**kwarg)
        trainer.train()
    elif args["task"] == 'finetune':
        trainer = FTrainer(**kwarg)
        trainer.train()
    else:  # inference
        trainer = Tester(**kwarg)
        trainer.test()


if __name__ == '__main__':
    main()
