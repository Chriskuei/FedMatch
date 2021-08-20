# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import copy
import random
import json5
import torch
from datetime import datetime
from pprint import pformat
from collections import OrderedDict

from .utils.loader import load_data
from .utils.logger import Logger
from .utils.params import validate_params
from .model import Model
from .interface import Interface


class Trainer:
    def __init__(self, args):
        self.args = args
        self.log = Logger(self.args, 0)

    def train(self):
        start_time = datetime.now()
        server, interface, states = self.build_model()

        clients = []
        train_datasets = []
        dev_datasets = []
        logs = []
        for i, dataset in enumerate(self.args.data_dir):
            clients.append(Model(dataset=dataset, args=self.args, number=i))
            train = load_data(dataset, 'train')
            dev = load_data(dataset, self.args.eval_file)
            train_datasets.append(train)
            dev_datasets.append(dev)
            logs.append(Logger(self.args, i))
            self.log(f'{dataset} train ({len(train)}) | {self.args.eval_file} ({len(dev)})')

        train_batches = []
        for i, dataset in enumerate(train_datasets):
            train_batches.append(interface.pre_process(dataset, i))
        dev_batches = []
        for i, dataset in enumerate(dev_datasets):
            dev_batches.append(interface.pre_process(dataset, i, training=False))
        self.log('setup complete: {}s.'.format(str(datetime.now() - start_time).split(".")[0]))

        if self.args.sample == 'sqrt' or self.args.sample == 'prop':
            probs = [len(batches) for batches in train_batches]
            alpha = 0
            if self.args.sample == 'prop':
                alpha = 1.
            if self.args.sample == 'sqrt':
                alpha = 0.5
            probs = [p ** alpha for p in probs]
            tot = sum(probs)
            probs = [p/tot for p in probs]
        
        aggregation_weights = [1. / len(clients) for client in clients]

        try:
            for fed_round in range(self.args.round):
                self.log(f'training in round {fed_round}.')
                for client, train, dev, log, i in zip(clients, train_batches, dev_batches, logs, range(len(train_datasets))):
                    server_params = server.network.state_dict()
                    client_params = client.network.state_dict()
                    share_params = {}

                    if self.args.fed_type == 'fedpatch':
                        for key, data in server_params.items():
                            if 'prediction' in key:
                                share_params[key] = copy.deepcopy(data)
                            else:
                                share_params[key] = copy.deepcopy(client_params[key])
                    elif self.args.fed_type == 'dual_bert':
                        for key, data in server_params.items():
                            if 'bert_keep' not in key and 'prediction' not in key:
                                share_params[key] = copy.deepcopy(data)
                            else:
                                share_params[key] = copy.deepcopy(client_params[key])
                    elif self.args.fed_type == 'LayerNorm':
                        for key, data in server_params.items():
                            if 'LayerNorm' not in key and 'prediction' not in key:
                                share_params[key] = copy.deepcopy(data)
                            else:
                                share_params[key] = copy.deepcopy(client_params[key])
                    elif self.args.fed_type == 'diff_privacy' or self.args.fed_type == 'median':
                        def dp_noise(param, sigma):
                            noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                            return noised_layer
                        for key, data in server_params.items():
                            data.add_(dp_noise(data, self.args.sigma))
                            share_params[key] = copy.deepcopy(data)
                    elif self.args.fed_type == 'fedavg':
                        for key, data in server_params.items():
                            share_params[key] = copy.deepcopy(data)
                    # elif self.args.fed_type == 'median':
                    #     for key, data in server_params.items():
                    #         share_params[key] = copy.deepcopy(data)
                    elif self.args.fed_type == 'self-center':
                        for key, data in server_params.items():
                            share_params[key] = (copy.deepcopy(data).cpu() * len(clients) - copy.deepcopy(client_params[key]).cpu()) / (len(clients)-1) * 0.5 + copy.deepcopy(client_params[key]).cpu() * 0.5
                    elif self.args.fed_type == 'pals':
                        for key, data in server_params.items():
                            if 'aug' in key or 'predictions' in key or 'mult' in key or 'gamma' in key or 'beta' in key:
                                share_params[key] = copy.deepcopy(client_params[key])
                            else:
                                share_params[key] = copy.deepcopy(data)
                    elif self.args.fed_type == 'fed_vertical':
                        for key, data in server_params.items():
                            if 'aug' in key or 'predictions' in key or 'mult' in key or 'vertical' in key or 'gamma' in key or 'beta' in key:
                                share_params[key] = copy.deepcopy(client_params[key])
                            else:
                                share_params[key] = copy.deepcopy(data)
                    client.network.load_state_dict(share_params)
                    for epoch in range(states['start_epoch'], self.args.epochs[i] + 1):
                        epoch = fed_round * self.args.epochs[i] + epoch
                        states['epoch'] = epoch
                        log.set_epoch(epoch)
                        if self.args.sample == 'anneal':
                            probs = [len(batches) for batches in train_batches]
                            alpha = 1. - 0.8 * epoch / (self.args.epochs[i] * self.args.round - 1)
                            print(alpha)
                            probs = [p**alpha for p in probs]
                            tot = sum(probs)
                            probs = [p/tot for p in probs]
                        if self.args.sample != 'all':
                            train = random.sample(train, k=int(len(train) * probs[i]))
                        batches = interface.shuffle_batch(train, i)
                        for batch_id, batch in enumerate(batches):
                            stats = client.update(batch, origin=share_params)
                            log.update(stats)
                            eval_per_updates = self.args.eval_per_updates[i] \
                                if client.updates > self.args.eval_warmup_steps[i] else self.args.eval_per_updates_warmup[i]
                            if client.updates % eval_per_updates == 0 or (self.args.eval_epoch and batch_id + 1 == len(batches)):
                                log.newline()
                                score, dev_stats = client.evaluate(dev)
                                if score > states['best_eval']:
                                    states['best_eval'], states['best_epoch'], states['best_step'], states['best_stats'] = score, epoch, client.updates, dev_stats
                                    if self.args.save:
                                        client.save(states, name=client.best_model_name)
                                log.log_eval(dev_stats)
                                if self.args.save_all:
                                    client.save(states)
                                    client.save(states, name='last')
                                if client.updates - states['best_step'] > self.args.early_stopping[i] \
                                        and client.updates > self.args.min_steps[i]:
                                    log('[Tolerance reached. Training is stopped early.]')
                                    raise EarlyStop('[Tolerance reached. Training is stopped early.]')
                            if stats['loss'] > self.args.max_loss:
                                raise EarlyStop('[Loss exceeds tolerance. Unstable training is stopped early.]')
                            if stats['lr'] < self.args.min_lr - 1e-6:
                                raise EarlyStop('[Learning rate has decayed below min_lr. Training is stopped early.]')
                        log.newline()
                    client.network.to('cpu')
                
                # Aggregation
                update_state = OrderedDict()
                if self.args.fed_type == 'median':
                    for k, client in enumerate(clients):
                        local_state = client.network.state_dict()
                        for key in server.network.state_dict().keys():
                            if k == 0:
                                update_state[
                                    key] = copy.deepcopy(local_state[key].unsqueeze(dim=-1))
                            else:
                                update_state[
                                    key] = torch.cat(
                                        (update_state[key], local_state[key].unsqueeze(dim=-1)),
                                        dim=-1
                                    )
                    for key in update_state.keys():
                        update_state[key] = update_state[key].median(dim=-1).values
                else:
                    for k, client in enumerate(clients):
                        local_state = client.network.state_dict()
                        for key in server.network.state_dict().keys():
                            if k == 0:
                                update_state[
                                    key] = local_state[key] * aggregation_weights[k]
                            else:
                                update_state[
                                    key] += local_state[key] * aggregation_weights[k]
                server.network.load_state_dict(update_state)

            self.log('Training complete.')
        except KeyboardInterrupt:
            self.log.newline()
            self.log(f'Training interrupted. Stopped early.')
        except EarlyStop as e:
            self.log.newline()
            self.log(str(e))
        self.log(f'best dev score {states["best_eval"]} at step {states["best_step"]} '
                 f'(epoch {states["best_epoch"]}).')
        self.log(f'best eval stats [{self.log.best_eval_str}]')
        training_time = str(datetime.now() - start_time).split('.')[0]
        self.log(f'Training time: {training_time}.')
        states['start_time'] = str(start_time).split('.')[0]
        states['training_time'] = training_time
        return states

    def build_model(self):
        states = {}
        interface = Interface(self.args, self.log)
        self.log(f'#classes: {self.args.num_classes}; #vocab: {self.args.num_vocab}')
        if self.args.seed:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.cuda:
                torch.cuda.manual_seed(self.args.seed)
            if self.args.deterministic:
                torch.backends.cudnn.deterministic = True

        model = Model(dataset='server', args=self.args)

        # set initial states
        states['start_epoch'] = 1
        states['best_eval'] = 0.
        states['best_epoch'] = 0
        states['best_step'] = 0

        self.log(f'trainable params: {model.num_parameters():,d}')
        self.log(f'trainable params (exclude embeddings): {model.num_parameters(exclude_embed=True):,d}')
        validate_params(self.args)
        with open(os.path.join(self.args.summary_dir, 'args.json5'), 'w') as f:
            json5.dump(self.args.__dict__, f, indent=2)
        self.log(pformat(vars(self.args), indent=2, width=120))
        return model, interface, states


class EarlyStop(Exception):
    pass
