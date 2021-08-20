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
import sys
import json5
import requests
from pprint import pprint

from src.utils import params
from src.trainer import Trainer


def main():
    argv = sys.argv
    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])
        for args, config in arg_groups:
            trainer = Trainer(args)
            states = trainer.train()
            with open('models/log.jsonl', 'a') as f:
                f.write(json5.dumps({
                    'params': config,
                    'state': states,
                }))
                f.write('\n')
    elif len(argv) == 3 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print('Usage: "python train.py configs/xxx.json5"')


if __name__ == '__main__':
    main()
