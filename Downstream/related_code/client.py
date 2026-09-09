import os
import json
import flwr as fl
from collections import OrderedDict
import torch
import random
from prepare import prepare_dataset, prepare_model, fix_seed
import Model.util.misc as misc

'''
随机数参数由client指定，开始训练时约定一个random seed，此后每次都由该随机数生成分享的参数index
因为flwr的机制限制了，每轮的参数聚合只能依据client的传入内容

需要解决的问题：对于中途加入的同一个seed的client，如何同步随机数生成到哪一步了（通过server传递的config指定第几个随机数）
'''
seed = 0    # to be set as the cmd argument later
INFO = '[FLWR CLIENT INFO]'


class UWFClient(fl.client.NumPyClient):
    def __init__(
            self, is_pretrain, 
            save_model_func,
            #seed,
            args=None, 
            epoch=0, 
            **kwargs
            ):
        '''
        Self-supervised pretraining if is_pretrain=True, else supervised finetuning.
        '''
        super(UWFClient, self).__init__(**kwargs)
        self.is_pretrain = is_pretrain
        self.save_model = save_model_func
        self.args = args
        
        self.epoch = epoch
        self.max_auc = 0.0
        
        # partial weight sharing
        #import random
        #random.seed(seed)
        self.indices = []
        self.args.partial_weight_fraction = 0.2 # to be set as the cmd argument later

        self.prepare(args)
        
    def prepare(self, args):
        misc.init_distributed_mode(args)
        
        self.device = torch.device(args.device)
        # prepare data and model, as well as random seed
        fix_seed(args)
        data_dict = prepare_dataset(args)
        model, model_without_ddp, n_parameters, criterion, optimizer, loss_scaler = prepare_model(args)
        
        if args.pretrain:
            from Model.pretrain import train_one_epoch
            evaluate = None
        else:
            from Model.finetune import train_one_epoch, evaluate
        
        self.data_dict = data_dict
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.n_parameters = n_parameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.train_func, self.test = train_one_epoch, evaluate

        self.param_num = self.model.state_dict().__len__()
        random.seed(seed)
      

    def get_parameters(self, config=None):
        '''
        return the model weight as a list of NumPy ndarrays
        '''
        print(INFO, 'send parameters to server (with the shared_weight_id:)')
        if 'round' not in config:
            self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        else:
            random.seed(seed)
            for i in range(config['round']):
                self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        print(self.shared_weight_id)
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        #print([params[id].shape for id in self.shared_weight_id])
        return [params[id] for id in self.shared_weight_id]


    def set_parameters(self, parameters, config=None):
        '''
        update the local model weights with the parameters received from the server
        '''
        if 'round' in config:
            random.seed(seed)
            for i in range(config['round']):
                self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        else:
            self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        # 每次从server接收参数时，只接受一部分参数，而不是全部参数
        print(INFO, 'set parameters from server (with the shared_weight_id:)')
        print(self.shared_weight_id)
        # params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # self.model.load_state_dict(state_dict, strict=True)

        # for i, id in enumerate(self.shared_weight_id):
        #     self.model.state_dict()[list(self.model.state_dict().keys())[id]].data = torch.tensor(parameters[i], device=self.device)
        state_dict = {}
        key_list = list(self.model.state_dict().keys())
        for i,id in enumerate(key_list):
            if i in self.shared_weight_id:
                #print(i, id, self.model.state_dict()[id].data.shape, torch.tensor(parameters[self.shared_weight_id.index(i)]).shape)
                state_dict[id] = torch.tensor(parameters[self.shared_weight_id.index(i)])
            else:
                state_dict[id] = self.model.state_dict()[id].data
        state_dict = OrderedDict(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        print(INFO, 'parameters successfully set!')


    def fit(self, parameters, config=None):
        '''
        set the local model weights
        train the local model
        receive the updated local model weights
        '''
        if self.epoch >= self.args.epochs:
            exit(0)
        print(INFO, len(parameters), config)

        # random.seed(seed)
        # for i in range(config['round']):
        #     self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        # self.set_parameters(parameters)
        
        print(INFO, 'Begin to train one round...')
        
            
        train_stats = self.train_func(
                model = self.model,
                optimizer = self.optimizer,
                loss_scaler = self.loss_scaler,
                criterion = self.criterion,
                data_dict = self.data_dict,
                device = self.device,
                current_epoch = self.epoch,
                args = self.args,
            )
        
        if self.data_dict["log_writer"] is not None:
            self.data_dict["log_writer"].flush()
        # save to log.txt
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': self.epoch,
                        'n_parameters': self.n_parameters}
        with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        # evaluate
        self.evaluate()
        
        print(INFO, 'Training one round finished!')
        self.epoch = self.epoch + 1
        # metrics: {lr: float, loss: float}
        print(len(self.data_dict["data_loader_train"]))
        return self.get_parameters(config=config), len(self.data_dict["data_loader_train"]), train_stats 
        #return self.get_partial_parameters(), len(self.data_dict["data_loader_train"]), train_stats 


    def evaluate(self, parameters=None, config=None):
        '''
        test the local model
        '''
        print(INFO, 'begin to evaluate')
        if parameters is not None:
            print(INFO, len(parameters), config)
            self.set_parameters(parameters, config)
        if self.is_pretrain:
            if self.args.output_dir and (self.epoch % 5 == 0 or self.epoch + 1 == self.args.epochs):
                self.save_model(
                    args = self.args,
                    model = self.model,
                    model_without_ddp = self.model_without_ddp,
                    optimizer = self.optimizer,
                    loss_scaler = self.loss_scaler,
                    epoch = self.epoch,
                    save_dir = self.args.output_dir
                )
            loss = 0.0
            len_dataset = len(self.data_dict["data_loader_train"])
            metrics = {"accuracy": 1}
        else:
            val_stats, val_auc_roc = self.test(
                                        model = self.model,
                                        data_dict = self.data_dict,
                                        device = self.device,
                                        task = self.args.task,
                                        mode = "val",
                                        num_class = self.args.num_classes,
                                        multi_label = self.args.multi_label
                                    )
            if val_auc_roc > self.max_auc:
                self.max_auc = val_auc_roc
                if self.args.output_dir:
                    print("Trying to save model ...")
                    self.save_model(
                        args = self.args,
                        model = self.model,
                        model_without_ddp = self.model_without_ddp,
                        optimizer = self.optimizer,
                        loss_scaler = self.loss_scaler,
                        epoch = self.epoch,
                        save_dir = self.args.output_dir
                    )
                    print("Model successfully saved!")
            if self.data_dict["log_writer"] is not None:
                self.data_dict["log_writer"].add_scalar('perf/val_acc1', val_stats['acc1'], self.epoch)
                self.data_dict["log_writer"].add_scalar('perf/val_auc', val_auc_roc, self.epoch)
                self.data_dict["log_writer"].add_scalar('perf/val_loss', val_stats['loss'], self.epoch)

            loss = float(val_stats['loss'])
            accuracy = val_stats['acc1']
            len_dataset = len(self.data_dict["data_loader_test"])
            metrics = {"accuracy": float(accuracy), 'auc_roc': float(val_auc_roc)}
        
        print(INFO, 'evaluation finished')
        return loss, len_dataset, metrics


class _UWFClient(fl.client.NumPyClient):
    def __init__(
            self, is_pretrain, 
            save_model_func,
            #seed,
            args=None, 
            epoch=0, 
            **kwargs
            ):
        '''
        Self-supervised pretraining if is_pretrain=True, else supervised finetuning.
        '''
        super(UWFClient, self).__init__(**kwargs)
        self.is_pretrain = is_pretrain
        self.save_model = save_model_func
        self.args = args
        
        self.epoch = epoch
        self.max_auc = 0.0
        
        # partial weight sharing
        #import random
        #random.seed(seed)
        self.indices = []
        self.args.partial_weight_fraction = 0.2 # to be set as the cmd argument later

        self.prepare(args)
        
    def prepare(self, args):
        #misc.init_distributed_mode(args)
        self.device = torch.device(args.device)
        # prepare data and model, as well as random seed
        fix_seed(args)
        data_dict = prepare_dataset(args)
        model, model_without_ddp, n_parameters, criterion, optimizer, loss_scaler = prepare_model(args)
        
        if args.pretrain:
            from Model.pretrain import train_one_epoch
            evaluate = None
        else:
            from Model.finetune import train_one_epoch, evaluate
        
        self.data_dict = data_dict
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.n_parameters = n_parameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.train_func, self.test = train_one_epoch, evaluate

        self.param_num = self.model.state_dict().__len__()
        random.seed(seed)
      

    def get_parameters(self, config=None):
        '''
        return the model weight as a list of NumPy ndarrays
        '''
        print(INFO, 'send parameters to server (with the shared_weight_id:)')
        #####
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        #####
        if 'round' not in config:
            #self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
            self.shared_weight_id = range(self.param_num)
        else:
            random.seed(seed)
            for i in range(config['round']):
                self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        print(self.shared_weight_id)
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        #print([params[id].shape for id in self.shared_weight_id])
        return [params[id] for id in self.shared_weight_id]


    def set_parameters(self, parameters, config=None):
        '''
        update the local model weights with the parameters received from the server
        '''
        #####
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        print(INFO, 'parameters successfully set!')
        return
        #####
        if 'round' in config:
            random.seed(seed)
            for i in range(config['round']):
                self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        else:
            #self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
            self.shared_weight_id = range(self.param_num)
        # 每次从server接收参数时，只接受一部分参数，而不是全部参数
        print(INFO, 'set parameters from server (with the shared_weight_id:)')
        print(self.shared_weight_id)

        # for i, id in enumerate(self.shared_weight_id):
        #     self.model.state_dict()[list(self.model.state_dict().keys())[id]].data = torch.tensor(parameters[i], device=self.device)
        state_dict = {}
        key_list = list(self.model.state_dict().keys())
        for i,id in enumerate(key_list):
            if i in self.shared_weight_id:
                #print(i, id, self.model.state_dict()[id].data.shape, torch.tensor(parameters[self.shared_weight_id.index(i)]).shape)
                state_dict[id] = torch.tensor(parameters[self.shared_weight_id.index(i)])
            else:
                state_dict[id] = self.model.state_dict()[id].data
        state_dict = OrderedDict(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        print(INFO, 'parameters successfully set!')


    def fit(self, parameters, config=None):
        '''
        set the local model weights
        train the local model
        receive the updated local model weights
        '''
        if self.epoch >= self.args.epochs:
            exit(0)
        print(INFO, len(parameters), config)

        # random.seed(seed)
        # for i in range(config['round']):
        #     self.shared_weight_id = random.sample(range(self.param_num), int(self.param_num * self.args.partial_weight_fraction))
        # self.set_parameters(parameters)
        
        print(INFO, 'Begin to train one round...')
        
            
        train_stats = self.train_func(
                model = self.model,
                optimizer = self.optimizer,
                loss_scaler = self.loss_scaler,
                criterion = self.criterion,
                data_dict = self.data_dict,
                device = self.device,
                current_epoch = self.epoch,
                args = self.args,
            )
        
        if self.data_dict["log_writer"] is not None:
            self.data_dict["log_writer"].flush()
        # save to log.txt
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': self.epoch,
                        'n_parameters': self.n_parameters}
        with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        # evaluate
        self.evaluate()
        
        print(INFO, 'Training one round finished!')
        self.epoch = self.epoch + 1
        # metrics: {lr: float, loss: float}
        return self.get_parameters(config=config), len(self.data_dict["data_loader_train"]), train_stats 
        #return self.get_partial_parameters(), len(self.data_dict["data_loader_train"]), train_stats 


    def evaluate(self, parameters=None, config=None):
        '''
        test the local model
        '''
        print(INFO, 'begin to evaluate')
        if parameters is not None:
            print(INFO, len(parameters), config)
            self.set_parameters(parameters, config)
        if self.is_pretrain:
            if self.args.output_dir and (self.epoch % 5 == 0 or self.epoch + 1 == self.args.epochs):
                self.save_model(
                    args = self.args,
                    model = self.model,
                    model_without_ddp = self.model_without_ddp,
                    optimizer = self.optimizer,
                    loss_scaler = self.loss_scaler,
                    epoch = self.epoch,
                    save_dir = self.args.output_dir
                )
            loss = 0.0
            len_dataset = len(self.data_dict["data_loader_train"])
            metrics = {"accuracy": 1}
        else:
            val_stats, val_auc_roc = self.test(
                                        model = self.model,
                                        data_dict = self.data_dict,
                                        device = self.device,
                                        task = self.args.task,
                                        mode = "val",
                                        num_class = self.args.num_classes,
                                        multi_label = self.args.multi_label
                                    )
            if val_auc_roc > self.max_auc:
                self.max_auc = val_auc_roc
                if self.args.output_dir:
                    print("Trying to save model ...")
                    self.save_model(
                        args = self.args,
                        model = self.model,
                        model_without_ddp = self.model_without_ddp,
                        optimizer = self.optimizer,
                        loss_scaler = self.loss_scaler,
                        epoch = self.epoch,
                        save_dir = self.args.output_dir
                    )
                    print("Model successfully saved!")
            if self.data_dict["log_writer"] is not None:
                self.data_dict["log_writer"].add_scalar('perf/val_acc1', val_stats['acc1'], self.epoch)
                self.data_dict["log_writer"].add_scalar('perf/val_auc', val_auc_roc, self.epoch)
                self.data_dict["log_writer"].add_scalar('perf/val_loss', val_stats['loss'], self.epoch)

            loss = float(val_stats['loss'])
            accuracy = val_stats['acc1']
            len_dataset = len(self.data_dict["data_loader_test"])
            metrics = {"accuracy": float(accuracy), 'auc_roc': float(val_auc_roc)}
        
        print(INFO, 'evaluation finished')
        return loss, len_dataset, metrics



if __name__ == '__main__':
    uwfclient = UWFClient(is_pretrain=True)
    fl.client.start_numpy_client(
        #server_address="frp-fit.top:65530",
        server_address="localhost:8080",
        client=uwfclient,
        grpc_max_message_length=0x7fffffff,
    )