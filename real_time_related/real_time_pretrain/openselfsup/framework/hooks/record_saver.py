from abc import ABCMeta, abstractmethod
import numpy as np
from bson.objectid import ObjectId
import datetime
import numpy as np
import collections
import inspect
import os
import os.path as osp
import copy
import time
import re
import sys
import threading
import git
import pdb
import pkg_resources
import json
from six import string_types
import torch
from argparse import Namespace

from ..utils import mkdir_or_exist
from .hook import Hook
from ..dist_utils import master_only


class Sonifier:
    def __init__(self, log, skip):
        self.log = log
        self.skip = skip
        self.git_cache = {}

    def version_info(self, module):
        """Get version of a standard python module.

        Args:
            module (module): python module object to get version info for.

        Returns:
            dict: dictionary of version info.

        """
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        else:
            pkgname = module.__name__.split('.')[0]
            try:
                info = pkg_resources.get_distribution(pkgname)
            except (pkg_resources.DistributionNotFound, pkg_resources.RequirementParseError):
                version = None
                self.log.warning(
                    'version information not found for %s -- what package is this from?' % module.__name__)
            else:
                version = info.version

        return {'version': version}

    def git_info(self, repo):
        """Return information about a git repo.

        Args:
            repo (git.Repo): The git repo to be investigated.

        Returns:
            dict: Git repo information

        """
        if repo.git_dir in self.git_cache:
            return self.git_cache[repo.git_dir]
        if repo.is_dirty():
            self.log.warning('repo %s is dirty -- having committment issues?' %
                        repo.git_dir)
            clean = False
        else:
            clean = True
        branchname = repo.active_branch.name
        commit = repo.active_branch.commit.hexsha
        origin = repo.remote('origin')
        urls = list(map(str, list(origin.urls)))
        remote_ref = [_r for _r in origin.refs if _r.name ==
                      'origin/' + branchname]
        if not len(remote_ref) > 0:
            self.log.warning('Active branch %s not in origin ref' % branchname)
            active_branch_in_origin = False
            commit_in_log = False
        else:
            active_branch_in_origin = True
            remote_ref = remote_ref[0]
            gitlog = remote_ref.log()
            shas = [_r.oldhexsha for _r in gitlog] + \
                [_r.newhexsha for _r in gitlog]
            if commit not in shas:
                self.log.warning(
                        'Commit %s not in remote origin log for branch %s' \
                                % (commit, branchname))
                commit_in_log = False
            else:
                commit_in_log = True
        info = {'git_dir': repo.git_dir,
                'active_branch': branchname,
                'commit': commit,
                'remote_urls': urls,
                'clean': clean,
                'active_branch_in_origin': active_branch_in_origin,
                'commit_in_log': commit_in_log}
        self.git_cache[repo.git_dir] = info
        return info

    def version_check_and_info(self, module):
        """Return either git info or standard module version if not a git repo.

        Args:
            module (module): python module object to get info for.

        Returns:
            dict: dictionary of info

        """
        srcpath = inspect.getsourcefile(module)
        try:
            repo = git.Repo(srcpath, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            log.info('module %s not in a git repo, checking package version' %
                     module.__name__)
            info = self.version_info(module)
        else:
            info = self.git_info(repo)
        info['source_path'] = srcpath
        return info

    def sonify(self, arg):
        """Return version of arg that can be trivally serialized to json format.

        Args:
            arg (object): an argument to sonify.
            memo (dict, optional): A dictionary to contain args. Defaults to None.
            skip (bool, optional): Skip git repo info check. Defaults to False.

        Returns:
            Sonified return argument.

        Raises:
            TypeError: Cannot sonify argument type.

        """
        if id(arg) in self.memo:
            rval = self.memo[id(arg)]

        if isinstance(arg, ObjectId):
            rval = arg
        elif isinstance(arg, torch.Tensor):
            # get data from variable
            try:
                # if variable is on GPU
                rval = arg.detach().cpu().item()
            except Exception:
                rval = arg.detach().item()
            rval = self.sonify(rval)
        elif isinstance(arg, datetime.datetime):
            rval = arg
        elif isinstance(arg, np.floating):
            rval = float(arg)
        elif isinstance(arg, np.integer):
            rval = int(arg)
        elif isinstance(arg, (list, tuple)):
            rval = type(arg)([self.sonify(ai) for ai in arg])
        elif isinstance(arg, collections.OrderedDict):
            rval = collections.OrderedDict([(self.sonify(k),
                                             self.sonify(v))
                                            for k, v in arg.items()])
        elif isinstance(arg, dict):
            rval = dict([(self.sonify(k), self.sonify(v))
                         for k, v in arg.items()])
        elif isinstance(arg, (string_types, float, int, type(None))):
            rval = arg
        elif isinstance(arg, np.ndarray):
            if arg.ndim == 0:
                rval = self.sonify(arg.sum())
            else:
                rval = list(map(self.sonify, arg))  # N.B. memo None
        # -- put this after ndarray because ndarray not hashable
        elif arg in (True, False):
            rval = int(arg)
        elif callable(arg):
            mod = inspect.getmodule(arg)
            modname = mod.__name__
            objname = arg.__name__
            if not self.skip:
                rval = self.version_check_and_info(mod)
            else:
                rval = {}
            rval.update({'objname': objname,
                         'modname': modname})
            rval = self.sonify(rval)
        elif isinstance(arg, Namespace):
            rval = self.sonify(vars(arg))
        else:
            raise TypeError('sonify', arg)

        self.memo[id(rval)] = rval
        return rval

    def __call__(self, arg):
        self.memo = {}
        return self.sonify(arg)


class BaseSaver(Hook):
    RESERVED_KEYS = ['params', 'step', 'epoch']
    def __init__(
            self, params, logger, interval=1, 
            by_epoch=True, skip_check=False):
        self.logger = logger
        self.sonifier = Sonifier(log=logger, skip=skip_check)
        logger.info('Logging params to records')
        self.sonified_params = self.sonifier(params)
        logger.info('Params logged')
        self.by_epoch = by_epoch
        self.interval = interval

    def before_run(self, runner):
        if len(self.records_to_save) > 1:
            self.dump(runner, post_step=False)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        to_save = self.every_n_epochs(runner, self.interval)
        if to_save:
            self.dump(runner)

    def after_train_iter(self, runner):
        if self.by_epoch:
            return
        to_save = self.every_n_iters(runner, self.interval)
        if to_save:
            self.dump(runner)

    def init_records(self):
        self.records_to_save = {
                'params': self.sonified_params}

    def add_epoch_step(self, runner, post_step=True):
        self.records_to_save['step'] = runner.iter + int(post_step)
        self.records_to_save['epoch'] = runner.epoch + int(post_step)

    @master_only
    def save(self, record):
        assert isinstance(record, dict), "Record must be a dict"

        all_rec = self.records_to_save
        for key, value in record.items():
            assert not key in self.RESERVED_KEYS, \
                    "Key cannot be in reserved keys of record saver"
            if key not in all_rec:
                all_rec[key] = []
            all_rec[key].append(self.sonifier(value))

    def dump(self, runner, post_step=True):
        raise NotImplementedError


class FileSaver(BaseSaver):
    def __init__(self, out_dir, *args, **kwargs):
        super(FileSaver, self).__init__(*args, **kwargs)
        self.out_dir = osp.join(out_dir, 'records')
        mkdir_or_exist(self.out_dir)

        self.init_records()

    @master_only
    def dump(self, runner, post_step=True):
        self.add_epoch_step(runner, post_step)
        rec = self.records_to_save
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_name = 'epoch{}_iter{}_time{}.json'.format(
                runner.epoch, runner.iter, timestamp)
        with open(osp.join(self.out_dir, json_name), 'w') as fout:
            json.dump(rec, fout)
        self.init_records()


def make_mongo_safe(_d):
    """Make a json-izable actually safe for insertion into Mongo.

    Args:
        _d (dict): a dictionary to make safe for Mongo.

    """
    klist = list(_d.keys())
    for _k in klist:
        if hasattr(_d[_k], 'keys'):
            make_mongo_safe(_d[_k])
        if not isinstance(_k, str):
            _d[str(_k)] = _d.pop(_k)
        _k = str(_k)
        if '.' in _k:
            _d[_k.replace('.', '___')] = _d.pop(_k)


class MongoDBSaver(BaseSaver):
    def __init__(
            self, 
            port,
            database_name,
            collection_name,
            exp_id,
            host='localhost',
            *args, **kwargs):
        import pymongo as pm
        super(MongoDBSaver, self).__init__(*args, **kwargs)
        self.init_records()

        self.host = host
        self.port = port
        self.database_name = database_name
        self.collection_name = collection_name
        self.exp_id = exp_id

        self.checkpoint_threads = []
        self.client = pm.MongoClient(self.host, self.port)
        self.database = self.client[self.database_name]

        if self.collection_name not in self.database.collection_names():
            self.collection = self.database[self.collection_name]
            self.collection.create_index('exp_id')
        else:
            self.collection = self.database[self.collection_name]

        self.RESERVED_KEYS.append('exp_id')

    def _close(self):
        self.client.close()

    @master_only
    def dump(self, runner, post_step=True):
        self.add_epoch_step(runner, post_step)
        rec = self.records_to_save
        make_mongo_safe(rec)
        rec['exp_id'] = self.exp_id
        self.collection.save(rec)
        self.init_records()
